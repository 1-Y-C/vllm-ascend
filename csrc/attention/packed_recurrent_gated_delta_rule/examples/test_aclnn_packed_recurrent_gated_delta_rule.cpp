#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_packed_recurrent_gated_delta_rule.h"

namespace {

#define LOG_PRINT(fmt, ...) (void)printf(fmt, ##__VA_ARGS__)

int64_t ShapeSize(const std::vector<int64_t>& shape) {
  int64_t s = 1;
  for (auto d : shape) s *= d;
  return s;
}

int32_t Init(int32_t deviceId, aclrtStream* stream) {
  int32_t ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) { LOG_PRINT("aclInit failed: %d\n", ret); return ret; }
  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return ret; }
  ret = aclrtCreateStream(stream);
  if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return ret; }
  return ACL_SUCCESS;
}

int32_t MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void** devAddr, aclTensor** tensor) {
  auto size = ShapeSize(shape) * (aclDataTypeSize(dtype));
  auto ret = aclrtMalloc(devAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) { LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret; }
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--)
    strides[i] = shape[i + 1] * strides[i + 1];
  *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                            ACL_FORMAT_ND, shape.data(), shape.size(), *devAddr);
  return ACL_SUCCESS;
}

struct TestResources {
  void* mixedQkvDev{nullptr};
  void* aDev{nullptr};
  void* bDev{nullptr};
  void* aLogDev{nullptr};
  void* dtBiasDev{nullptr};
  void* stateDev{nullptr};
  void* ssmIndicesDev{nullptr};
  void* attnOutDev{nullptr};
  void* workspace{nullptr};
  aclTensor* mixedQkvTensor{nullptr};
  aclTensor* aTensor{nullptr};
  aclTensor* bTensor{nullptr};
  aclTensor* aLogTensor{nullptr};
  aclTensor* dtBiasTensor{nullptr};
  aclTensor* stateTensor{nullptr};
  aclTensor* ssmIndicesTensor{nullptr};
  aclTensor* attnOutTensor{nullptr};
};

void Cleanup(TestResources& res, aclrtStream stream, int32_t devId) {
  auto destroy = [](aclTensor* t) { if (t) aclDestroyTensor(t); };
  auto free_ = [](void* p) { if (p) aclrtFree(p); };
  destroy(res.mixedQkvTensor); destroy(res.aTensor); destroy(res.bTensor);
  destroy(res.aLogTensor); destroy(res.dtBiasTensor); destroy(res.stateTensor);
  destroy(res.ssmIndicesTensor); destroy(res.attnOutTensor);
  free_(res.mixedQkvDev); free_(res.aDev); free_(res.bDev);
  free_(res.aLogDev); free_(res.dtBiasDev); free_(res.stateDev);
  free_(res.ssmIndicesDev); free_(res.attnOutDev); free_(res.workspace);
  if (stream) aclrtDestroyStream(stream);
  aclrtResetDevice(devId);
  aclFinalize();
}

} // namespace

int32_t main() {
  const int32_t deviceId = 0;
  const int64_t B = 1, HV = 1, DV = 128;
  const float scale = 0.088388f;

  aclrtStream stream = nullptr;
  TestResources res = {};
  uint64_t wsSize = 0;
  aclOpExecutor* executor = nullptr;

  // 1. Init
  if (Init(deviceId, &stream) != ACL_SUCCESS) { return 1; }

  // 2. Shapes
  std::vector<int64_t> mixedQkvShape = {B, 3 * HV * DV};
  std::vector<int64_t> abShape = {B, HV, DV};
  std::vector<int64_t> logBiasShape = {HV};
  std::vector<int64_t> stateShape = {B, HV, DV, DV};
  std::vector<int64_t> ssmIdxShape = {B, 1};
  std::vector<int64_t> outShape = {B, 1, HV, DV};

  // 3. Create tensors
  #define CHECK_TENSOR(call) do { int32_t _r = (call); if (_r != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; } } while(0)
  CHECK_TENSOR(MakeTensor(mixedQkvShape, ACL_BF16, &res.mixedQkvDev, &res.mixedQkvTensor));
  CHECK_TENSOR(MakeTensor(abShape, ACL_BF16, &res.aDev, &res.aTensor));
  CHECK_TENSOR(MakeTensor(abShape, ACL_BF16, &res.bDev, &res.bTensor));
  CHECK_TENSOR(MakeTensor(logBiasShape, ACL_FLOAT, &res.aLogDev, &res.aLogTensor));
  CHECK_TENSOR(MakeTensor(logBiasShape, ACL_FLOAT, &res.dtBiasDev, &res.dtBiasTensor));
  CHECK_TENSOR(MakeTensor(stateShape, ACL_BF16, &res.stateDev, &res.stateTensor));
  CHECK_TENSOR(MakeTensor(ssmIdxShape, ACL_INT32, &res.ssmIndicesDev, &res.ssmIndicesTensor));
  CHECK_TENSOR(MakeTensor(outShape, ACL_BF16, &res.attnOutDev, &res.attnOutTensor));

  // 4. Init data and copy to device
  std::vector<uint16_t> mixedQkvH(ShapeSize(mixedQkvShape), 0x3F80);
  auto ret = aclrtMemcpy(res.mixedQkvDev, mixedQkvH.size() * sizeof(uint16_t),
                         mixedQkvH.data(), mixedQkvH.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  std::vector<uint16_t> abH(ShapeSize(abShape), 0x3F00);
  ret = aclrtMemcpy(res.aDev, abH.size() * sizeof(uint16_t), abH.data(), abH.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }
  ret = aclrtMemcpy(res.bDev, abH.size() * sizeof(uint16_t), abH.data(), abH.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  std::vector<float> aLogH(HV, 0.0f);
  ret = aclrtMemcpy(res.aLogDev, aLogH.size() * sizeof(float), aLogH.data(), aLogH.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  std::vector<float> dtBiasH(HV, 0.0f);
  ret = aclrtMemcpy(res.dtBiasDev, dtBiasH.size() * sizeof(float), dtBiasH.data(), dtBiasH.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  std::vector<uint16_t> stateH(ShapeSize(stateShape), 0);
  ret = aclrtMemcpy(res.stateDev, stateH.size() * sizeof(uint16_t), stateH.data(), stateH.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  std::vector<int32_t> ssmIdxH(B * 1, 0);
  ret = aclrtMemcpy(res.ssmIndicesDev, ssmIdxH.size() * sizeof(int32_t), ssmIdxH.data(), ssmIdxH.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) { Cleanup(res, stream, deviceId); return 1; }

  // 5. Get workspace
  ret = aclnnPackedRecurrentGatedDeltaRuleGetWorkspaceSize(
      res.mixedQkvTensor, res.aTensor, res.bTensor,
      res.aLogTensor, res.dtBiasTensor, res.stateTensor,
      res.ssmIndicesTensor, scale, res.attnOutTensor,
      &wsSize, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("GetWorkspaceSize failed: %d\n", ret);
    Cleanup(res, stream, deviceId);
    return 1;
  }

  if (wsSize > 0) {
    ret = aclrtMalloc(&res.workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("workspace malloc failed: %d\n", ret);
      Cleanup(res, stream, deviceId);
      return 1;
    }
  }

  // 6. Execute
  ret = aclnnPackedRecurrentGatedDeltaRule(res.workspace, wsSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("Execute failed: %d\n", ret);
    Cleanup(res, stream, deviceId);
    return 1;
  }

  // 7. Sync
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("Sync failed: %d\n", ret);
    Cleanup(res, stream, deviceId);
    return 1;
  }

  // 8. Read output
  std::vector<uint16_t> attnOutH(ShapeSize(outShape));
  ret = aclrtMemcpy(attnOutH.data(), attnOutH.size() * sizeof(uint16_t),
                    res.attnOutDev, attnOutH.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("D2H failed: %d\n", ret);
    Cleanup(res, stream, deviceId);
    return 1;
  }

  // 9. Validate: output should not be all zeros
  bool allZero = true;
  for (auto v : attnOutH) {
    if (v != 0) { allZero = false; break; }
  }
  LOG_PRINT("attnOut sample[0] = 0x%04X (bf16), allZero=%s\n",
            attnOutH[0], allZero ? "true" : "false");

  if (allZero) {
    LOG_PRINT("FAIL: attnOut is all zeros\n");
    Cleanup(res, stream, deviceId);
    return 1;
  }

  LOG_PRINT("PASS: packed_recurrent_gated_delta_rule punch-through test passed\n");
  Cleanup(res, stream, deviceId);
  return 0;
}

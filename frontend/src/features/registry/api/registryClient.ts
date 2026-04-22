import { opsClient } from "../../ops/api/opsClient";

export const registryClient = {
  getRegistry() {
    return opsClient.getRegistry();
  },
  activateModel(modelId: string) {
    return opsClient.activateModel(modelId);
  },
  rollbackModel() {
    return opsClient.rollbackModel();
  }
};

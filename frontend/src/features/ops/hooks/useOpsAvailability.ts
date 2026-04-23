const LOCAL_DEV_REASON = "Ops workspace is unavailable locally because no VITE_OPS_API_TOKEN is configured.";

export function useOpsAvailability() {
  const opsEnabledByEnv = import.meta.env.VITE_ENABLE_OPS !== "false";
  const token = import.meta.env.VITE_OPS_API_TOKEN?.trim();

  if (!opsEnabledByEnv) {
    return {
      opsAvailable: false,
      reason: "Ops workspace is disabled via VITE_ENABLE_OPS=false."
    };
  }

  if (import.meta.env.DEV && !token) {
    return {
      opsAvailable: false,
      reason: LOCAL_DEV_REASON
    };
  }

  return {
    opsAvailable: true,
    reason: null
  };
}

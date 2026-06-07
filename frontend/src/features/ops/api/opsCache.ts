export interface OpsCachedRequestOptions<T> {
  key: string;
  staleMs: number;
  force?: boolean;
  request: () => Promise<T>;
}

interface OpsCacheEntry<T> {
  value?: T;
  timestamp: number;
  inFlight?: Promise<T>;
}

const cache = new Map<string, OpsCacheEntry<unknown>>();

function now(): number {
  return Date.now();
}

export function peekOpsCache<T>(key: string, staleMs?: number): T | null {
  const entry = cache.get(key) as OpsCacheEntry<T> | undefined;
  if (!entry || entry.value === undefined) return null;
  if (staleMs !== undefined && now() - entry.timestamp >= staleMs) return null;
  return entry.value;
}

export async function cachedOpsRequest<T>({
  key,
  staleMs,
  force = false,
  request,
}: OpsCachedRequestOptions<T>): Promise<T> {
  const entry = cache.get(key) as OpsCacheEntry<T> | undefined;
  if (!force && entry?.value !== undefined && now() - entry.timestamp < staleMs) {
    return entry.value;
  }

  if (entry?.inFlight) {
    return entry.inFlight;
  }

  const nextEntry: OpsCacheEntry<T> = entry ?? { timestamp: 0 };
  const inFlight = request()
    .then((value) => {
      nextEntry.value = value;
      nextEntry.timestamp = now();
      return value;
    })
    .finally(() => {
      delete nextEntry.inFlight;
    });

  nextEntry.inFlight = inFlight;
  cache.set(key, nextEntry as OpsCacheEntry<unknown>);
  return inFlight;
}

export function invalidateOpsCache(keys?: string | string[]): void {
  if (!keys) {
    cache.clear();
    return;
  }

  const keyList = Array.isArray(keys) ? keys : [keys];
  for (const key of keyList) {
    cache.delete(key);
  }
}

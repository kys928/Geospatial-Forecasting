const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function resolveApiBaseUrl(envValue: string | undefined): string {
  const configuredBaseUrl = envValue?.trim();
  return configuredBaseUrl ? configuredBaseUrl : DEFAULT_API_BASE_URL;
}

const API_BASE_URL = resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL);

export class HttpError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data: unknown) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.data = data;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const contentType = response.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");
  const responseData = isJson ? ((await response.json()) as unknown) : await response.text();

  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    if (typeof responseData === "object" && responseData && "detail" in responseData) {
      message = `${message} - ${String((responseData as { detail: unknown }).detail)}`;
    }
    throw new HttpError(message, response.status, responseData);
  }

  return responseData as T;
}

export async function httpGet<T>(path: string, headers?: HeadersInit): Promise<T> {
  return requestJson<T>(path, {
    method: "GET",
    headers
  });
}

export async function httpPost<TResponse, TBody = unknown>(
  path: string,
  body?: TBody,
  headers?: HeadersInit
): Promise<TResponse> {
  return requestJson<TResponse>(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...headers
    },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
}

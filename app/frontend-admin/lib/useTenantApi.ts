"use client";

import { useCallback } from "react";
import apiClient from "@/lib/apiClient";
import { useTenant } from "@/contexts/TenantContext";
import type { AxiosRequestConfig, AxiosResponse } from "axios";

/**
 * Tenant-aware API helper hook.
 *
 * Wraps apiClient with automatic tenant injection for common patterns:
 *
 * - `get(url, config)` → adds `client_id` to query params
 * - `postJson(url, data, config)` → adds `client_id` to JSON body
 * - `postForm(url, formData, config)` → adds `business_id` to FormData
 *
 * For APIs that use different param names (e.g. `tenant_id` in query),
 * use the existing API route helpers in `apiRoutes.ts` directly.
 *
 * @example
 * const api = useTenantApi();
 *
 * // GET with client_id in query
 * const res = await api.get("/api/v2/retrieve/intents");
 *
 * // POST JSON with client_id in body
 * const res = await api.postJson("/api/v2/retrieve/query", { query: "..." });
 *
 * // POST FormData with business_id
 * const form = new FormData();
 * form.append("file", file);
 * const res = await api.postForm("/api/v2/ingestion/upload", form);
 */
export function useTenantApi() {
  const { clientId } = useTenant();

  /**
   * GET request with client_id injected into query params.
   */
  const get = useCallback(
    <T = any>(
      url: string,
      config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> => {
      const params = {
        ...(config?.params || {}),
        client_id: clientId,
      };
      return apiClient.get<T>(url, { ...config, params });
    },
    [clientId]
  );

  /**
   * POST JSON request with client_id injected into request body.
   */
  const postJson = useCallback(
    <T = any>(
      url: string,
      data?: Record<string, any>,
      config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> => {
      const body = {
        ...(data || {}),
        client_id: clientId,
      };
      return apiClient.post<T>(url, body, config);
    },
    [clientId]
  );

  /**
   * POST FormData request with business_id appended.
   * Use for file uploads where the backend expects business_id in form.
   */
  const postForm = useCallback(
    <T = any>(
      url: string,
      formData: FormData,
      config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> => {
      // Append business_id if not already present
      if (!formData.has("business_id")) {
        formData.append("business_id", clientId);
      }
      return apiClient.post<T>(url, formData, {
        ...config,
        headers: {
          ...(config?.headers || {}),
          "Content-Type": "multipart/form-data",
        },
      });
    },
    [clientId]
  );

  /**
   * PUT JSON request with client_id injected into request body.
   */
  const putJson = useCallback(
    <T = any>(
      url: string,
      data?: Record<string, any>,
      config?: AxiosRequestConfig
    ): Promise<AxiosResponse<T>> => {
      const body = {
        ...(data || {}),
        tenant_id: clientId,
      };
      return apiClient.put<T>(url, body, config);
    },
    [clientId]
  );

  return {
    clientId,
    get,
    postJson,
    postForm,
    putJson,
    // Expose raw client for non-tenant-scoped calls
    raw: apiClient,
  };
}

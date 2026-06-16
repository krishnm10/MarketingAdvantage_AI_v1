// frontend-admin/lib/apiClient.ts

import axios from "axios";
import { clearAuthToken } from "./authToken";

const apiClient = axios.create({
  baseURL: "/api/proxy",
  timeout: 30000,
  withCredentials: true,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      clearAuthToken();

      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("mai:unauthorized"));
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;

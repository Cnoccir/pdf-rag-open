// src/api/health.ts

import axios from 'axios'; // Import directly from axios package

// Create a dedicated axios instance for health API
const healthApi = axios.create({
  baseURL: '/api'
});

/**
 * Service for interacting with the health monitoring API
 */
export const health = {
  /**
   * Get system health information
   */
  getSystemHealth: async () => {
    return healthApi.get('/health/system');
  },

  /**
   * Get database health information
   */
  getDatabaseHealth: async () => {
    return healthApi.get('/health/databases');
  },

  /**
   * Get vector store health information
   */
  getVectorStoreHealth: async () => {
    return healthApi.get('/health/vector_stores');
  },

  /**
   * Get memory health information
   */
  getMemoryHealth: async () => {
    return healthApi.get('/health/memory');
  },

  /**
   * Get system-wide metrics
   */
  getMetrics: async () => {
    return healthApi.get('/health/metrics');
  },

  /**
   * Get health information for a specific PDF
   */
  getPdfHealth: async (pdfId: string) => {
    if (!pdfId) {
      throw new Error('PDF ID is required');
    }
    return healthApi.get(`/health/pdf/${pdfId}`);
  },

  /**
   * Run a test query against a specific PDF
   */
  runTestQuery: async (query: string, pdfId: string, k: number = 3) => {
    if (!query) {
      throw new Error('Query text is required');
    }
    if (!pdfId) {
      throw new Error('PDF ID is required');
    }
    return healthApi.post('/health/query_test', {
      query,
      pdf_id: pdfId,
      k
    });
  },

  /**
   * Force reinitialization of vector stores
   */
  reinitializeVectorStores: async () => {
    return healthApi.get('/health/databases?force_init=true');
  },

  /**
   * Run a comprehensive system diagnostic
   */
  runDiagnostic: async () => {
    return healthApi.get('/health/diagnostic');
  }
};

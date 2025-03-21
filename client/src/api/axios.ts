import axios from 'axios';
import { addError } from '$s/errors';

// Create consistent axios instance with proper error handling
export const api = axios.create({
  baseURL: '/api'  // Keep as is, this is correct
});

// Enhanced request logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, config.data || '');
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);
// Enhanced response logging with detailed error handling
api.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} from ${response.config.url}`, response.data || '');
    return response;
  },
  (err) => {
    // Extract useful error details
    if (err.response) {
      const { status, data, config } = err.response;
      console.error(`[API Error] ${status} from ${config.url}`, data || err.message);

      // Handle server errors by adding to error store
      if (status >= 500) {
        addError({
          contentType: err.response.headers['Content-Type'] || err.response.headers['content-type'],
          message: getErrorMessage(err)
        });
      }
    } else if (err.request) {
      console.error('[API Network Error] No response received', err.request);
    } else {
      console.error('[API Error] Request configuration error', err.message);
    }
    return Promise.reject(err);
  }
);
// Trigger embedding process for a document
export async function triggerEmbeddingProcess(pdfId: string): Promise<void> {
    try {
        const response = await api.post(`/documents/${pdfId}/trigger-embedding`);
        if (response.status !== 200) {
            throw new Error('Failed to trigger embedding process.');
        }
    } catch (error) {
        console.error(`Error triggering embedding for document ID ${pdfId}:`, error);
        const errorMessage = getErrorMessage(error);
        addError({
            message: errorMessage
        });
        throw new Error(errorMessage);
    }
}

export const deletePdf = async (pdfId: string) => {
  try {
    const response = await axiosInstance.delete(`/pdfs/${pdfId}/delete`);
    return response.data;
  } catch (error) {
    console.error('Error deleting PDF:', error);
    throw error;
  }
};

// Improved error message extraction
export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const apiError = error.response?.data;

    // Handle string errors
    if (typeof apiError === 'string' && apiError.length > 0) {
      return apiError;
    }

    // Handle object errors
    if (apiError && typeof apiError === 'object') {
      return apiError.message || apiError.error || error.message;
    }

    // Fallback to status text or axios message
    return error.response?.statusText || error.message;
  }

  // Handle non-axios errors
  if (error instanceof Error) {
    return error.message;
  }

  // Handle unknown error types
  return 'Unknown error occurred. Please try again.';
};

export const getError = (error: unknown) => {
	if (axios.isAxiosError(error)) {
		const apiError = error.response?.data as ApiError;
		return apiError;
	}

	return null;
};

export async function activateResearchMode(conversationId: string): Promise<void> {
    try {
        const response = await api.post(`/conversations/${conversationId}/research/activate`);
        if (response.status !== 200) {
            throw new Error('Failed to activate research mode');
        }
    } catch (error) {
        console.error('Error activating research mode:', error);
        const errorMessage = getErrorMessage(error);
        addError({
            message: errorMessage
        });
        throw new Error(errorMessage);
    }
}

export async function deactivateResearchMode(conversationId: string): Promise<void> {
    try {
        const response = await api.post(`/conversations/${conversationId}/research/deactivate`);
        if (response.status !== 200) {
            throw new Error('Failed to deactivate research mode');
        }
    } catch (error) {
        console.error('Error deactivating research mode:', error);
        const errorMessage = getErrorMessage(error);
        addError({
            message: errorMessage
        });
        throw new Error(errorMessage);
    }
}
// Export the health API
export { health };

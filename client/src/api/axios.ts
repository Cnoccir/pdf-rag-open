import axios from 'axios';
import { addError } from '$s/errors';

interface ApiError {
	message: string;
	error: string;
}

export const api = axios.create({
	baseURL: '/api'
});

api.interceptors.response.use(
	(res) => res,
	(err) => {
		if (err.response && err.response.status >= 500) {
			const { response } = err;
			const message = getErrorMessage(err);

			if (message) {
				addError({
					contentType: response.headers['Content-Type'] || response.headers['content-type'],
					message: getErrorMessage(err)
				});
			}
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

export const getErrorMessage = (error: unknown) => {
	if (axios.isAxiosError(error)) {
		const apiError = error.response?.data as ApiError;
		if (typeof apiError === 'string' && (apiError as string).length > 0) {
			return apiError;
		}
		return apiError?.message || apiError?.error || error.message;
	}

	if (error instanceof Error) {
		return error.message;
	}

	if (
		error &&
		typeof error === 'object' &&
		'message' in error &&
		typeof error.message === 'string'
	) {
		return error.message;
	}

	return 'Something went wrong';
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

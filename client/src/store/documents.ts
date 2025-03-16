import { writable } from 'svelte/store';
import { api, getErrorMessage } from '$api';

export interface Document {
	id: string;
	file_id?: string;
	name: string;
	processed?: boolean;
	error?: string;
	description?: string;
	category?: string;
	created_at?: string;
	updated_at?: string;
	metadata?: any;
}

interface UploadStore {
	data: Document[];
	error: string;
	uploadProgress: number;
}

const INITIAL_STATE: UploadStore = {
	data: [],
	error: '',
	uploadProgress: 0
};

// This is what your original code is importing, so we must keep the same name
export const documents = writable<UploadStore>(INITIAL_STATE);

const set = (val: Partial<UploadStore>) => {
	documents.update((state) => ({ ...state, ...val }));
};

const setUploadProgress = (event: ProgressEvent) => {
	const progress = Math.round((event.loaded / event.total) * 100);
	set({ uploadProgress: progress });
};

// This is what your original code is importing, so we must keep the same name
export const upload = async (file: File) => {
	set({ error: '' });

	try {
		const formData = new FormData();
		formData.append('file', file);

		await api.post('/pdfs', formData, {
			onUploadProgress: setUploadProgress
		});
	} catch (error) {
		return set({ error: getErrorMessage(error) });
	}
};

// This is what your original code is importing, so we must keep the same name
export const clearErrors = () => {
	set({ error: '', uploadProgress: 0 });
};

export const getDocuments = async () => {
	set({ error: '' });

	try {
		const { data } = await api.get('/pdfs');
		set({ data });
		return data;
	} catch (error) {
		set({ error: getErrorMessage(error) });
		return null;
	}
};

// Add new delete document function
export const deleteDocument = async (id: string) => {
	set({ error: '' });

	try {
		await api.delete(`/pdfs/${id}`);

		// Update the store by removing the deleted document
		documents.update(state => ({
			...state,
			data: state.data.filter(doc => doc.id !== id)
		}));

		return true;
	} catch (error) {
		set({ error: getErrorMessage(error) });
		return false;
	}
};

// Add new re-trigger embedding function
export const triggerEmbedding = async (id: string) => {
	set({ error: '' });

	try {
		await api.post(`/pdfs/${id}/trigger-embedding`);
		return true;
	} catch (error) {
		set({ error: getErrorMessage(error) });
		return false;
	}
};

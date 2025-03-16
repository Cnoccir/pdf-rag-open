import axios from 'axios';  // If you're using the axios library

export async function triggerEmbeddingProcess(pdfId: string) {
    const response = await fetch(`/api/pdfs/${pdfId}/trigger-embedding`, {  // Changed from /embed
        method: 'POST'
    });

    if (!response.ok) {
        throw new Error('Failed to trigger embedding process');
    }

    return await response.json();
}

export async function softDeleteDocument(pdfId: string) {
    try {
        const response = await fetch(`/api/pdfs/${pdfId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to delete document');
        }

        return await response.json();
    } catch (error) {
        console.error('Delete error:', error);
        throw error;
    }
}


export async function getPdfContent(pdfId: string) {
    const response = await fetch(`/api/pdfs/${pdfId}/content`);

    if (!response.ok) {
        throw new Error('Failed to fetch PDF content');
    }

    return await response.blob();
}

export async function uploadDocument(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/pdfs', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Failed to upload document');
    }

    return await response.json();
}

export async function getDocuments() {
    const response = await fetch('/api/pdfs');

    if (!response.ok) {
        throw new Error('Failed to fetch documents');
    }

    return await response.json();
}

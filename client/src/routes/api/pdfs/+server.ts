import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

interface Document {
  id: string;
  name: string;
  description?: string;
  category?: string;
  updatedAt: string;
  url: string;
}

export const GET: RequestHandler = async ({ url }) => {
    try {
        // Fetch documents from your existing backend
        const response = await fetch('http://localhost:5000/api/pdfs');
        const documents: Document[] = await response.json();
        
        // Add default values for new fields if they don't exist
        const enhancedDocuments = documents.map(doc => ({
            ...doc,
            description: doc.description || '',
            category: doc.category || 'general',
            updatedAt: doc.updatedAt || new Date().toISOString(),
            url: `/api/pdfs/${doc.id}/content`
        }));
        
        return json(enhancedDocuments);
    } catch (error) {
        console.error('Error fetching documents:', error);
        return json({ error: 'Failed to fetch documents' }, { status: 500 });
    }
};

export const POST: RequestHandler = async ({ request }) => {
    try {
        const formData = await request.formData();
        const file = formData.get('file');
        const category = formData.get('category') || 'general';
        const description = formData.get('description') || '';
        
        if (!file) {
            return json({ error: 'No file provided' }, { status: 400 });
        }

        // Create new FormData with additional fields
        const newFormData = new FormData();
        newFormData.append('file', file);
        newFormData.append('category', category as string);
        newFormData.append('description', description as string);

        // Forward to your existing backend
        const uploadResponse = await fetch('http://localhost:5000/api/pdfs', {
            method: 'POST',
            body: newFormData
        });

        const result = await uploadResponse.json();
        return json(result);
    } catch (error) {
        console.error('Error creating document:', error);
        return json({ error: 'Failed to create document' }, { status: 500 });
    }
};

export const DELETE: RequestHandler = async ({ url }) => {
    try {
        const id = url.searchParams.get('id');
        
        if (!id) {
            return json({ error: 'No document ID provided' }, { status: 400 });
        }

        // Forward delete request to your existing backend
        const deleteResponse = await fetch(`http://localhost:5000/api/pdfs/${id}`, {
            method: 'DELETE'
        });

        const result = await deleteResponse.json();
        return json(result);
    } catch (error) {
        console.error('Error deleting document:', error);
        return json({ error: 'Failed to delete document' }, { status: 500 });
    }
};

// Add endpoint for fetching PDF content
export const GET_CONTENT: RequestHandler = async ({ params }) => {
    try {
        const { id } = params;
        const response = await fetch(`http://localhost:5000/api/pdfs/${id}/content`);
        const pdfBuffer = await response.arrayBuffer();
        
        return new Response(pdfBuffer, {
            headers: {
                'Content-Type': 'application/pdf',
                'Content-Disposition': 'inline'
            }
        });
    } catch (error) {
        console.error('Error fetching PDF content:', error);
        return json({ error: 'Failed to fetch PDF content' }, { status: 500 });
    }
};

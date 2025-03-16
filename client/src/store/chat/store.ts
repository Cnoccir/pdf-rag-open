import { get } from 'svelte/store';
import { writable } from '../writeable';
import { api } from '$api';

export interface Message {
	id?: number;
	role: 'user' | 'assistant' | 'system' | 'pending';
	content: string;
	metadata?: any; // Include metadata for research mode info
}

export interface Conversation {
	id: number;
	messages: Message[];
	pdf_id?: number; // Document ID
	metadata?: {
		research_mode?: {
			active: boolean;
			pdf_ids: string[];
			document_names?: Record<string, string>;
		}
	};
}

// Document interfaces for research mode
export interface DocumentInfo {
	id: string;
	name: string;
	concepts?: string[];
	confidence?: number;
	isPrimary?: boolean;
}

export interface MessageOpts {
	useStreaming?: boolean;
	useResearch?: boolean;
	documentId?: string;
	activeDocs?: string[]; // Add active document IDs
}

export interface ChatState {
	error: string;
	loading: boolean;
	activeConversationId: number | null;
	conversations: Conversation[];
	// Research mode fields
	researchMode: boolean;
	recommendedDocuments: DocumentInfo[];
	activeDocuments: DocumentInfo[];
	availableDocuments: DocumentInfo[];
	// Add this new field to track failed messages
	lastMessageFailed: boolean;
}

const INITIAL_STATE: ChatState = {
	error: '',
	loading: false,
	activeConversationId: null,
	conversations: [],
	// Research mode initial state
	researchMode: false,
	recommendedDocuments: [],
	activeDocuments: [],
	availableDocuments: [],
	// Initialize lastMessageFailed
	lastMessageFailed: false
};

const store = writable<ChatState>(INITIAL_STATE);

const set = (val: Partial<ChatState>) => {
	store.update((state) => ({ ...state, ...val }));
};

const getRawMessages = () => {
	const conversation = getActiveConversation();
	if (!conversation) {
		return [];
	}

	return conversation.messages
		.filter((message) => message.role !== 'pending')
		.map((message) => {
			return { role: message.role, content: message.content };
		});
};

const getActiveConversation = () => {
	const { conversations, activeConversationId } = get(store);
	if (!activeConversationId) {
		return null;
	}

	return conversations.find((c) => c.id === activeConversationId);
};

const insertMessageToActive = (message: Message) => {
	store.update((s) => {
		const conv = s.conversations.find((c) => c.id === s.activeConversationId);
		if (!conv) {
			return s;
		}
		conv.messages.push(message);
		return s;
	});
};

const removeMessageFromActive = (id: number) => {
	store.update((s) => {
		const conv = s.conversations.find((c) => c.id === s.activeConversationId);
		if (!conv) {
			return s;
		}
		conv.messages = conv.messages.filter((m) => m.id != id);
		return s;
	});
};

const scoreConversation = async (score: number) => {
	const conversationId = get(store).activeConversationId;

	return api.post(`/scores?conversation_id=${conversationId}`, { score });
};

/**
 * Extract recommended documents from the latest assistant message
 * to show document recommendations to the user
 */
const extractRecommendedDocuments = () => {
  const conversation = getActiveConversation();
  if (!conversation) return;

  // Find the last assistant message with recommendations
  const assistantMessages = conversation.messages.filter(
    m => m.role === 'assistant' && m.metadata?.recommended_documents
  );

  if (assistantMessages.length > 0) {
    const lastMessage = assistantMessages[assistantMessages.length - 1];
    const recommendedDocs = lastMessage.metadata?.recommended_documents || [];

    // Update recommended documents in store
    set({
      recommendedDocuments: recommendedDocs
    });

    console.log("Updated recommended documents:", recommendedDocs);
  }
};

const fetchConversations = async (documentId: number) => {
	set({ loading: true, error: '', lastMessageFailed: false });

	try {
		const { data } = await api.get<Conversation[]>(`/conversations?pdf_id=${documentId}`);

		// Process conversations to extract research mode info
		const processedConversations = data.map(conversation => {
			// Extract research mode status from conversation metadata if present
			let isResearchActive = false;
			if (conversation.metadata && conversation.metadata.research_mode) {
				isResearchActive = conversation.metadata.research_mode.active;
			}

			return conversation;
		});

		if (processedConversations.length) {
			set({
				conversations: processedConversations,
				activeConversationId: processedConversations[0].id,
				loading: false
			});

			// Set research mode if the active conversation has it enabled
			updateResearchModeFromActiveConversation();
		} else {
			await createConversation(documentId);
		}
	} catch (error) {
		console.error("Error fetching conversations:", error);
		set({
			error: error.response?.data?.error || 'Failed to load conversations',
			loading: false,
			lastMessageFailed: true
		});
	}
};

const createConversation = async (documentId: number) => {
	set({ loading: true, error: '', lastMessageFailed: false });

	try {
		const { data } = await api.post<Conversation>(`/conversations?pdf_id=${documentId}`);

		set({
			activeConversationId: data.id,
			conversations: [data, ...get(store).conversations],
			loading: false,
			// Reset research mode for new conversation
			researchMode: false,
			recommendedDocuments: [],
			activeDocuments: []
		});

		return data;
	} catch (error) {
		console.error("Error creating conversation:", error);
		set({
			error: error.response?.data?.error || 'Failed to create conversation',
			loading: false,
			lastMessageFailed: true
		});
		return null;
	}
};

const setActiveConversationId = (id: number) => {
	set({ activeConversationId: id, lastMessageFailed: false });

	// Update research mode state based on the newly selected conversation
	updateResearchModeFromActiveConversation();
};

const updateResearchModeFromActiveConversation = () => {
  const conversation = getActiveConversation();
  if (!conversation) return;

  // Check if conversation has research mode metadata
  if (conversation.metadata && conversation.metadata.research_mode) {
    const researchMode = conversation.metadata.research_mode;

    // CRITICAL FIX: Ensure we check both active flag AND multiple PDFs
    if (researchMode.active === true && researchMode.pdf_ids && researchMode.pdf_ids.length > 1) {
      // Extract document names and active documents
      let activeDocuments = researchMode.pdf_ids
        .filter(id => id) // Filter out null/undefined IDs
        .map(id => {
          // Get document name from metadata if available
          const name = researchMode.document_names?.[id] || `Document ${id}`;

          // Check if this is a valid document entry
          if (!id || (!name.includes('.') && name === `Document ${id}`)) {
            console.log(`Found potentially invalid document: ${id}, ${name}`);
            // Return null for invalid documents, we'll filter these out
            return null;
          }

          return {
            id: id.toString(),
            name: name,
            // Mark as primary if this is the primary document
            isPrimary: String(id) === String(conversation.pdf?.id)
          };
        })
        .filter(doc => doc !== null); // Remove invalid documents

      // Ensure primary document is included
      const primaryDocId = conversation.pdf?.id?.toString();
      const primaryDocName = conversation.pdf?.name;

      if (primaryDocId && !activeDocuments.some(doc => String(doc.id) === String(primaryDocId))) {
        activeDocuments.unshift({
          id: primaryDocId,
          name: primaryDocName || `Document ${primaryDocId}`,
          isPrimary: true
        });
      }

      // Filter out any documents with generic "Document" name and no proper ID
      activeDocuments = activeDocuments.filter(doc =>
        doc.name !== "Document" &&
        doc.id &&
        doc.id.length > 0
      );

      // CRITICAL: Only set research mode to true if we have active flag AND multiple docs
      const hasMultipleDocs = activeDocuments.length > 1;

      set({
        researchMode: researchMode.active && hasMultipleDocs,
        activeDocuments
      });

      console.log(`Set research mode to: ${researchMode.active && hasMultipleDocs} with ${activeDocuments.length} documents`);
    } else {
      set({
        researchMode: false,
        activeDocuments: []
      });
      console.log("Research mode deactivated: Missing active flag or not enough documents");
    }
  } else {
    set({
      researchMode: false,
      activeDocuments: []
    });
    console.log("No research_mode metadata found");
  }

  // Extract recommended documents from the last assistant message if available
  extractRecommendedDocuments();
};

// Updated function to send message with improved research mode support
const sendMessage = async (text: string, opts: MessageOpts = {}) => {
  const conversationId = get(store).activeConversationId;
  if (!conversationId) return null;

  set({ loading: true, error: '', lastMessageFailed: false });

  // CRITICAL FIX: Capture current research mode state before sending message
  const currentState = get(store);
  const isResearchActive = currentState.researchMode;
  const activeDocuments = currentState.activeDocuments.map(doc => doc.id);

  console.log(`Sending message with research mode: ${isResearchActive}, docs: ${activeDocuments.length}`);

  // Add user message and pending message immediately for better UX
  const userMessage: Message = {
    role: 'user',
    content: text
  };

  const pendingMessage: Message = {
    role: 'pending',
    content: ''
  };

  store.update(s => {
    const conversation = s.conversations.find(c => c.id === conversationId);
    if (conversation) {
      conversation.messages = [...conversation.messages, userMessage, pendingMessage];
    }
    return s;
  });

  try {
    // IMPORTANT: Explicitly set useResearch based on current state
    opts.useResearch = isResearchActive;
    opts.activeDocs = activeDocuments;

    const response = await api.post(`/conversations/${conversationId}/messages`, {
      input: text,
      useStreaming: opts.useStreaming || false,
      useResearch: isResearchActive, // Always use current state
      activeDocs: activeDocuments // Send active doc IDs
    });

    // Remove pending message
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
      }
      return s;
    });

    // Add the actual messages from the API
    const apiUserMessage = response.data.message;
    const assistantMessage = response.data.response;

    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages.push(apiUserMessage, assistantMessage);

        // Update conversation metadata if available
        if (assistantMessage.metadata && assistantMessage.metadata.research_mode) {
          conversation.metadata = {
            ...conversation.metadata,
            research_mode: assistantMessage.metadata.research_mode
          };

          console.log("Updated research mode in conversation metadata:",
                     assistantMessage.metadata.research_mode);
        }
      }
      return s;
    });

    // Update research mode status based on assistant response
    if (assistantMessage.metadata) {
      // Extract recommended documents
      const recommendedDocs = assistantMessage.metadata.recommended_documents || [];

      // Extract active documents from research mode
      let activeDocs: DocumentInfo[] = [];
      let isResearchActive = false;

      if (assistantMessage.metadata.research_mode) {
        const researchMode = assistantMessage.metadata.research_mode;

        // CRITICAL FIX: Validate research mode properly
        isResearchActive = researchMode.active === true &&
                          researchMode.pdf_ids &&
                          researchMode.pdf_ids.length > 1;

        if (isResearchActive && researchMode.pdf_ids && researchMode.document_names) {
          activeDocs = researchMode.pdf_ids.map(id => ({
            id,
            name: researchMode.document_names[id] || `Document ${id}`,
            isPrimary: id === String(apiUserMessage.conversation_id)
          }));
        }
      }

      set({
        recommendedDocuments: recommendedDocs,
        activeDocuments: activeDocs,
        researchMode: isResearchActive,
        loading: false,
        lastMessageFailed: false
      });

      console.log(`Updated UI state: research_mode=${isResearchActive}, active_docs=${activeDocs.length}`);
    }

    return { userMessage: apiUserMessage, assistantMessage };
  } catch (error) {
    console.error("Error sending message:", error);

    // Remove pending message on error
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
      }
      return s;
    });

    set({
      error: error.response?.data?.error || 'Failed to send message',
      loading: false,
      lastMessageFailed: true
    });

    return null;
  }
};

/**
 * Regenerate the response for a previously failed query with improved research mode handling
 * @param query The original query to regenerate a response for
 * @param options Options for message sending
 * @returns A Promise that resolves when the regeneration is complete
 */
const regenerateResponse = async (query: string, options: MessageOpts = {}) => {
    const conversationId = get(store).activeConversationId;
    if (!conversationId) {
        throw new Error('No active conversation');
    }

    set({ loading: true, error: '', lastMessageFailed: false });

    try {
        // Get conversation from store
        const conversation = getActiveConversation();
        if (!conversation) {
            throw new Error('Could not find active conversation');
        }

        // CRITICAL FIX: Capture current research mode state before sending message
        const currentState = get(store);
        const isResearchActive = currentState.researchMode;
        const activeDocuments = currentState.activeDocuments.map(doc => doc.id);

        console.log(`Regenerating response with research mode: ${isResearchActive}, docs: ${activeDocuments.length}`);

        // Add regeneration header
        const headers = {
            'X-Regeneration-Request': 'true'
        };

        // IMPORTANT: Explicitly set research mode options
        options.useResearch = isResearchActive;
        options.activeDocs = activeDocuments;

        // Use the existing API endpoint with regeneration header
        const response = await api.post(
            `/conversations/${conversationId}/messages`,
            {
                input: query,
                useStreaming: options.useStreaming || false,
                useResearch: isResearchActive, // Use current state
                activeDocs: activeDocuments // Send active doc IDs
            },
            { headers }
        );

        // Response should contain both user message and AI response
        const { message, response: assistantMessage } = response.data;

        if (!message || !assistantMessage) {
            throw new Error('Invalid response from server');
        }

        // Update the conversation - for regeneration, replace existing messages
        store.update(s => {
            const conversation = s.conversations.find(c => c.id === conversationId);
            if (!conversation) return s;

            // Remove pending or error messages
            conversation.messages = conversation.messages.filter(m =>
                m.role !== 'pending' &&
                !(m.role === 'assistant' && m.metadata?.status === 'error')
            );

            // Update conversation metadata if needed
            if (assistantMessage.metadata && assistantMessage.metadata.research_mode) {
                conversation.metadata = {
                    ...conversation.metadata,
                    research_mode: assistantMessage.metadata.research_mode
                };

                console.log("Updated research mode in metadata during regeneration:",
                          assistantMessage.metadata.research_mode);
            }

            // Add the new response
            conversation.messages.push(assistantMessage);

            return s;
        });

        // Update research mode status based on assistant response
        if (assistantMessage.metadata) {
            // Extract recommended documents
            const recommendedDocs = assistantMessage.metadata.recommended_documents || [];

            // Extract active documents from research mode
            let activeDocs: DocumentInfo[] = [];
            let isResearchActive = false;

            if (assistantMessage.metadata.research_mode) {
                const researchMode = assistantMessage.metadata.research_mode;

                // CRITICAL FIX: Validate research mode properly
                isResearchActive = researchMode.active === true &&
                                  researchMode.pdf_ids &&
                                  researchMode.pdf_ids.length > 1;

                if (isResearchActive && researchMode.pdf_ids && researchMode.document_names) {
                    activeDocs = researchMode.pdf_ids.map(id => ({
                        id,
                        name: researchMode.document_names[id] || `Document ${id}`,
                        isPrimary: id === String(message.conversation_id)
                    }));
                }
            }

            set({
                recommendedDocuments: recommendedDocs,
                activeDocuments: activeDocs,
                researchMode: isResearchActive,
                loading: false,
                lastMessageFailed: false
            });

            console.log(`Updated UI state after regeneration: research_mode=${isResearchActive}, active_docs=${activeDocs.length}`);
        } else {
            set({ loading: false, lastMessageFailed: false });
        }

        return { message, assistantMessage };
    } catch (error) {
        console.error('Failed to regenerate response:', error);

        set({
            loading: false,
            error: error.response?.data?.error || 'Failed to regenerate response',
            lastMessageFailed: true
        });

        throw error;
    }
};

// Updated activateResearchMode function with improved structure
const activateResearchMode = async (conversationId: number, pdfIds: string[]) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    console.log("Activating research mode with PDF IDs:", pdfIds);

    // Make sure all IDs are strings and filter out empty values
    const formattedPdfIds = pdfIds
      .map(id => id.toString())
      .filter(id => id.trim().length > 0);

    if (formattedPdfIds.length < 2) {
      throw new Error("At least two valid PDF IDs are required for research mode");
    }

    // Get current conversation to identify primary document
    const currentState = get(store);
    const activeConv = currentState.conversations.find(c => c.id === conversationId);
    const primaryDocId = activeConv?.pdf_id?.toString();

    console.log("Primary document:", primaryDocId);

    // Ensure primary document is included
    if (primaryDocId && !formattedPdfIds.includes(primaryDocId)) {
      formattedPdfIds.unshift(primaryDocId);
    }

    // Get document names
    const documentsWithNames = await fetchDocumentNames(formattedPdfIds);

    console.log("Enhanced documents for research:", documentsWithNames);

    // CRITICAL FIX: Always use POST method with proper data structure
    const response = await api.post(`/conversations/${conversationId}/research/activate`, {
      pdf_ids: formattedPdfIds,
      active: true  // Explicitly set active flag
    });

    // Update store with properly named documents
    set({
      researchMode: true,
      activeDocuments: documentsWithNames,
      loading: false
    });

    // Update conversation metadata with proper structure
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        // Create a document_names mapping for metadata
        const documentNames = {};
        documentsWithNames.forEach(doc => {
          documentNames[doc.id] = doc.name;
        });

        // Create fresh research_mode object with proper structure
        const updatedMetadata = {
          ...conversation.metadata,
          research_mode: {
            active: true,  // CRITICAL: Explicitly set to true
            pdf_ids: [...formattedPdfIds],
            document_names: { ...documentNames }
          }
        };

        // Also update active_pdf_ids for backwards compatibility
        updatedMetadata.active_pdf_ids = [...formattedPdfIds];

        // Set the updated metadata
        conversation.metadata = updatedMetadata;
      }
      return s;
    });

    // Double-check the update worked
    const updatedState = get(store);
    const updatedConv = updatedState.conversations.find(c => c.id === conversationId);
    console.log("Updated conversation metadata:", updatedConv?.metadata);

    return true;
  } catch (error) {
    console.error("Error activating research mode:", error);
    set({
      error: error.response?.data?.error || 'Failed to activate research mode',
      loading: false,
      lastMessageFailed: true
    });

    return false;
  }
};

// Updated deactivateResearchMode function
const deactivateResearchMode = async (conversationId: number) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // CRITICAL FIX: Always use POST method with explicit active=false
    const response = await api.post(`/conversations/${conversationId}/research/deactivate`, {
      active: false  // Explicitly set active flag to false
    });

    set({
      researchMode: false,
      activeDocuments: [],
      loading: false
    });

    // Update conversation metadata to reflect research mode status with proper structure
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation && conversation.metadata) {
        // Create a new metadata object to ensure state updates properly
        conversation.metadata = {
          ...conversation.metadata,
          research_mode: {
            active: false,  // CRITICAL: Explicitly set to false
            pdf_ids: [],
            document_names: {}
          },
          active_pdf_ids: [] // For backwards compatibility
        };
      }
      return s;
    });

    console.log("Deactivated research mode, updated metadata");

    return true;
  } catch (error) {
    console.error("Error deactivating research mode:", error);

    // Try GET method as fallback if POST fails
    try {
      console.warn("POST method failed, trying GET instead");
      await api.get(`/conversations/${conversationId}/research/deactivate`);

      // Update store state
      set({
        researchMode: false,
        activeDocuments: [],
        loading: false
      });

      // Update conversation metadata
      store.update(s => {
        const conversation = s.conversations.find(c => c.id === conversationId);
        if (conversation && conversation.metadata) {
          conversation.metadata = {
            ...conversation.metadata,
            research_mode: {
              active: false,
              pdf_ids: [],
              document_names: {}
            },
            active_pdf_ids: []
          };
        }
        return s;
      });

      return true;
    } catch (fallbackError) {
      console.error("Both POST and GET methods failed:", fallbackError);
      set({
        error: error.response?.data?.error || 'Failed to deactivate research mode',
        loading: false,
        lastMessageFailed: true
      });

      return false;
    }
  }
};

// Updated acceptRecommendedDocument function with improved research mode handling
const acceptRecommendedDocument = async (documentId: string) => {
  const state = get(store);
  const conversationId = state.activeConversationId;

  if (!conversationId) return false;

  // Create list of all document IDs (existing + new)
  const currentDocIds = state.activeDocuments.map(d => d.id);
  const allDocIds = [...new Set([...currentDocIds, documentId])];

  // Update locally first for better UX
  store.update(s => ({
    ...s,
    recommendedDocuments: s.recommendedDocuments.filter(d => d.id !== documentId),
    loading: true
  }));

  console.log(`Accepting recommended document ${documentId}, activating research mode with docs:`, allDocIds);

  // Activate research mode with the new document
  const success = await activateResearchMode(conversationId, allDocIds);

  // Ensure the UI state is updated correctly by re-reading from conversation metadata
  if (success) {
    updateResearchModeFromActiveConversation();
  }

  return success;
};

// New function to fetch available documents
const fetchAvailableDocuments = async () => {
  try {
    const { data } = await api.get('/pdfs/');

    // Format documents with proper interface
    const formattedDocs: DocumentInfo[] = data.map(doc => ({
      id: doc.id.toString(),
      name: doc.name || `Document ${doc.id}`
    }));

    // Store for future use
    set({ availableDocuments: formattedDocs });

    console.log(`Fetched ${formattedDocs.length} available documents`);
    return formattedDocs;
  } catch (error) {
    console.error("Failed to fetch available documents:", error);
    return [];
  }
};

// Helper function to fetch document names
const fetchDocumentNames = async (pdfIds: string[]): Promise<DocumentInfo[]> => {
  try {
    // First, try to get names from existing documents in the store
    const existingDocs = get(store).availableDocuments;
    const cachedConversation = getActiveConversation();
    const primaryPdf = cachedConversation?.pdf;

    console.log("Fetching document names for:", pdfIds);
    console.log("Primary PDF:", primaryPdf);
    console.log("Available documents cache:", existingDocs);

    // If we have cached documents, use those names first
    if (existingDocs.length > 0) {
      return pdfIds.map(id => {
        // Check if this is the primary document from conversation
        if (primaryPdf && String(primaryPdf.id) === String(id)) {
          return {
            id: id.toString(),
            name: primaryPdf.name || `Document ${id}`,
            isPrimary: true
          };
        }

        // Otherwise look in cached documents
        const doc = existingDocs.find(d => String(d.id) === String(id));
        return {
          id: id.toString(),
          name: doc?.name || `Document ${id}`,
          isPrimary: false
        };
      });
    }

    // If no cached documents, fetch them individually
    console.log("No cached documents, fetching individually");
    const docPromises = pdfIds.map(async id => {
      try {
        const { data } = await api.get(`/pdfs/${id}`);
        console.log(`Fetched document ${id}:`, data);
        return {
          id: id.toString(),
          name: data.name || data.title || `Document ${id}`,
          isPrimary: primaryPdf ? String(primaryPdf.id) === String(id) : false
        };
      } catch (error) {
        console.error(`Failed to fetch document ${id}:`, error);
        return {
          id: id.toString(),
          name: `Document ${id}`,
          isPrimary: primaryPdf ? String(primaryPdf.id) === String(id) : false
        };
      }
    });

    const results = await Promise.all(docPromises);
    console.log("Fetched document names:", results);
    return results;
  } catch (error) {
    console.error("Error fetching document names:", error);
    return pdfIds.map(id => ({
      id: id.toString(),
      name: `Document ${id}`,
      isPrimary: false
    }));
  }
};

const resetAll = () => {
  set(INITIAL_STATE);
};

const resetError = () => {
  set({ error: '', lastMessageFailed: false });
};

export {
  store,
  set,
  setActiveConversationId,
  getRawMessages,
  fetchConversations,
  resetAll,
  resetError,
  createConversation,
  getActiveConversation,
  insertMessageToActive,
  removeMessageFromActive,
  scoreConversation,
  // Research mode exports
  sendMessage,
  regenerateResponse,
  activateResearchMode,
  deactivateResearchMode,
  acceptRecommendedDocument,
  fetchAvailableDocuments,
  updateResearchModeFromActiveConversation
};

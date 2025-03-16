<!-- DocumentSelectorModal.svelte -->
<script lang="ts">
  import { createEventDispatcher, onMount, afterUpdate } from 'svelte';
  import { fade } from 'svelte/transition';
  import Icon from '$c/Icon.svelte';

  export let isOpen = false;
  export let conversation = null;
  export let availableDocuments = [];

  // Maximum number of documents that can be selected
  const MAX_SELECTABLE_DOCUMENTS = 5;

  const dispatch = createEventDispatcher();
  let selectedDocIds = [];
  let searchQuery = '';
  let primaryDocId = '';
  let primaryDocName = '';
  let selectionError = '';

  // Enhance each document with information about whether it's primary
  $: processedDocuments = availableDocuments.map(doc => ({
    ...doc,
    isPrimary: doc.id.toString() === primaryDocId,
    isSelected: selectedDocIds.includes(doc.id.toString())
  }));

  // Filter documents based on search
  $: filteredDocuments = searchQuery
    ? processedDocuments.filter(doc =>
        doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.id.toString().includes(searchQuery))
    : processedDocuments;

  // Calculate selection count for display - don't count the primary document
  $: selectionCount = new Set(
    selectedDocIds.filter(id => id !== primaryDocId)
  ).size;

  // Force redraw of document items when selection changes
  $: selectedDocIdsString = JSON.stringify(selectedDocIds);

  // Initialize when component mounts
  onMount(() => {
    console.log("DocumentSelector mounted");
    initializeSelection();
  });

  // Also initialize whenever modal opens or conversation changes
  $: if (isOpen || conversation) {
    initializeSelection();
  }

  // After every update, log the current state
  afterUpdate(() => {
    console.log("After update - selected doc IDs:", selectedDocIds);
    console.log("Selection count:", selectionCount);
  });

  // Function to initialize the selection state
  function initializeSelection() {
    if (conversation && conversation.pdf) {
      primaryDocId = conversation.pdf.id.toString();
      primaryDocName = conversation.pdf.name || `Document ${primaryDocId}`;
      console.log("Primary document:", primaryDocId, primaryDocName);

      // Always start with the primary document selected
      selectedDocIds = [primaryDocId];

      // Add any already selected documents from existing research mode
      if (conversation?.metadata?.research_mode?.pdf_ids) {
        let existing = conversation.metadata.research_mode.pdf_ids;
        console.log("Existing PDF IDs:", existing);

        if (Array.isArray(existing)) {
          // Ensure all IDs are strings and filter out duplicates
          existing = existing.map(id => id.toString());

          // Make sure we include the primary document
          if (!existing.includes(primaryDocId)) {
            existing.push(primaryDocId);
          }

          // Create a Set to hold unique IDs
          const uniqueIds = new Set(existing);

          // Convert back to array
          selectedDocIds = Array.from(uniqueIds);
          console.log("Initialized selectedDocIds:", selectedDocIds);
        }
      }

      selectionError = '';

      // Log the state to verify
      console.log("After initialization: primary=", primaryDocId, "selected=", selectedDocIds);
    }
  }

  function closeModal() {
    dispatch('close');
  }

  function handleBackdropClick(e) {
    // Only close if clicking directly on the backdrop, not its children
    if (e.target === e.currentTarget) {
      closeModal();
    }
  }

  function handleBackdropKeydown(e) {
    if (e.key === 'Escape') {
      closeModal();
    }
  }

  function handleActivateResearch() {
      console.log("Activating research with documents:", selectedDocIds);

      if (conversation) {
        // Check if we have any documents selected (besides the primary)
        const hasAdditionalDocs = selectedDocIds.some(id => id !== primaryDocId);

        // Always ensure primary document is included
        let documentsToActivate = [...selectedDocIds];
        if (!documentsToActivate.includes(primaryDocId)) {
          documentsToActivate.unshift(primaryDocId);
        }

        // Remove any duplicates
        documentsToActivate = [...new Set(documentsToActivate)];

        console.log("Final documents for research:", documentsToActivate);

        dispatch('activateResearch', {
          conversationId: conversation.id,
          documentIds: documentsToActivate
        });

        // Close the modal after successful activation
        closeModal();
      } else {
        // Show an error or notification that documents need to be selected
        selectionError = 'Please select at least one document to activate research mode';
      }
    }

  function toggleDocumentSelection(docId) {
    console.log("Toggle selection for doc:", docId);

    // If it's the primary document, don't allow toggling off
    if (docId === primaryDocId) {
      console.log("Can't toggle primary document");
      return;
    }

    const docIdStr = docId.toString();

    // Check if already selected
    const isCurrentlySelected = selectedDocIds.includes(docIdStr);

    if (isCurrentlySelected) {
      // Remove from selection
      console.log("Removing document from selection:", docIdStr);
      selectedDocIds = selectedDocIds.filter(id => id !== docIdStr);
      selectionError = '';
    } else {
      // Check if we would exceed maximum documents
      if (new Set(selectedDocIds).size >= MAX_SELECTABLE_DOCUMENTS) {
        selectionError = `Maximum of ${MAX_SELECTABLE_DOCUMENTS} documents can be selected`;
        return;
      }

      // Add to selection
      console.log("Adding document to selection:", docIdStr);
      selectedDocIds = [...selectedDocIds, docIdStr];
      selectionError = '';
    }

    // Force update by creating a new array
    selectedDocIds = [...selectedDocIds];
    console.log("Updated selectedDocIds:", selectedDocIds);
  }

  function isDocumentSelected(docId) {
    const docIdStr = docId.toString();
    return selectedDocIds.includes(docIdStr);
  }

  function isPrimaryDocument(docId) {
    const docIdStr = docId.toString();
    return docIdStr === primaryDocId;
  }
</script>

{#if isOpen}
<div
  class="modal-backdrop"
  transition:fade={{duration: 150}}
  on:click={handleBackdropClick}
  on:keydown={handleBackdropKeydown}
  tabindex="-1"
  role="dialog"
  aria-modal="true"
  aria-labelledby="document-selector-title"
>
  <div
    class="modal-content"
    role="document"
  >
    <div class="modal-header">
      <h3 id="document-selector-title">Select Documents for Research</h3>
      <button
        class="close-button"
        on:click={closeModal}
        aria-label="Close modal"
      >
        <svg viewBox="0 0 24 24" width="20" height="20">
          <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>

    <div class="modal-body">
      <p class="description">
        Select additional documents to include in your research. This will allow you to compare and analyze content across multiple documents.
      </p>

      {#if primaryDocId}
        <div class="primary-document-banner">
          <div class="flex items-center">
            <svg class="mr-2 text-blue-700" viewBox="0 0 24 24" width="16" height="16">
              <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M15 10l-4 4-2-2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span class="font-medium">Primary Document: {primaryDocName}</span>
          </div>
          <div class="text-xs text-blue-700 mt-1">
            This document is automatically included in research mode
          </div>
        </div>
      {/if}

      <div class="search-container">
        <svg class="search-icon" viewBox="0 0 24 24" width="18" height="18">
          <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <input
          type="text"
          placeholder="Search documents..."
          bind:value={searchQuery}
          class="search-input"
          aria-label="Search documents"
        />
      </div>

      {#if selectionError}
        <div class="error-message">
          <svg viewBox="0 0 24 24" width="16" height="16" class="mr-2">
            <circle cx="12" cy="12" r="10" fill="#FEE2E2" stroke="#B91C1C" stroke-width="2"/>
            <path d="M12 8v5M12 16h.01" stroke="#B91C1C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          {selectionError}
        </div>
      {/if}

      {#if filteredDocuments.length === 0}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" width="36" height="36">
            <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z" stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M17 21v-8H7v8M7 3v5h8" stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <p>No documents found matching your search.</p>
        </div>
      {:else}
        <div class="documents-list">
          {#each filteredDocuments as doc (doc.id)}
            {@const isPrimary = doc.isPrimary || isPrimaryDocument(doc.id)}
            {@const isSelected = doc.isSelected || isDocumentSelected(doc.id)}
            {#key `${doc.id}-${selectedDocIdsString}`}
              <!-- Document item -->
              <div
                class="document-item {isSelected ? 'selected' : ''} {isPrimary ? 'primary' : ''}"
                on:click={() => toggleDocumentSelection(doc.id)}
                on:keypress={(e) => e.key === 'Enter' && toggleDocumentSelection(doc.id)}
                tabindex="0"
                role="checkbox"
                aria-checked={isSelected}
              >
                <!-- Document Icon -->
                <div class="doc-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M14 2V8H20" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </div>

                <!-- Document info -->
                <div class="doc-info">
                  <div class="doc-name">{doc.name || `Document ${doc.id}`}</div>
                  <div class="doc-id">ID: {doc.id}</div>
                  {#if isPrimary}
                    <div class="primary-tag">PRIMARY</div>
                    <div class="text-xs text-blue-700 mt-1">This is your main document</div>
                  {/if}
                </div>

                <!-- Checkbox -->
                <div class="checkbox-wrapper">
                  <div class="checkbox {isSelected ? 'checked' : ''}">
                    {#if isSelected}
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M20 6L9 17L4 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                      </svg>
                    {/if}
                  </div>
                </div>
              </div>
            {/key}
          {/each}
        </div>
      {/if}
    </div>

    <div class="modal-footer">
      <div class="selection-summary">
        {selectionCount} additional document{selectionCount !== 1 ? 's' : ''} selected
        {#if MAX_SELECTABLE_DOCUMENTS}
          <span class="selection-max">(Max: {MAX_SELECTABLE_DOCUMENTS})</span>
        {/if}
      </div>
      <div class="button-group">
        <button class="cancel-button" on:click={closeModal}>Cancel</button>
        <button
          class="activate-button"
          on:click={handleActivateResearch}
        >
          Activate Research Mode
        </button>
      </div>
    </div>
  </div>
</div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(2px);
  }

  .modal-content {
    background-color: white;
    border-radius: 10px;
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    overflow: hidden;
  }

  .modal-header {
    padding: 16px 20px;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .modal-header h3 {
    font-size: 18px;
    font-weight: 600;
    color: #1a202c;
    margin: 0;
  }

  .close-button {
    background: none;
    border: none;
    color: #64748b;
    cursor: pointer;
    padding: 5px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.2s;
  }

  .close-button:hover {
    background-color: #f1f5f9;
    color: #334155;
  }

  .modal-body {
    padding: 20px;
    overflow-y: auto;
    flex: 1;
  }

  .description {
    margin: 0 0 16px 0;
    color: #4b5563;
    font-size: 14px;
    line-height: 1.5;
  }

  .primary-document-banner {
    background-color: #eef2ff;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    border-left: 4px solid #4f46e5;
  }

  .search-container {
    position: relative;
    margin-bottom: 16px;
  }

  .search-icon {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: #9ca3af;
    pointer-events: none;
  }

  .search-input {
    width: 100%;
    padding: 10px 10px 10px 40px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .search-input:focus {
    border-color: #4a63ee;
    box-shadow: 0 0 0 3px rgba(74, 99, 238, 0.1);
  }

  .error-message {
    background-color: #fee2e2;
    color: #b91c1c;
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 14px;
    display: flex;
    align-items: center;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 0;
    color: #6b7280;
  }

  .empty-state svg {
    color: #9ca3af;
    margin-bottom: 12px;
  }

  .empty-state p {
    font-size: 14px;
    text-align: center;
    margin: 0;
  }

  .documents-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .document-item {
    display: flex;
    align-items: center;
    padding: 10px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    background-color: #f9fafb;
  }

  .document-item:hover {
    background-color: #f3f4f6;
    border-color: #d1d5db;
  }

  .document-item.selected {
    background-color: #eef2ff;
    border-color: #c7d2fe;
  }

  .document-item.primary {
    background-color: #dbeafe;
    border-color: #93c5fd;
  }

  .doc-icon {
    flex-shrink: 0;
    margin-right: 12px;
  }

  .doc-info {
    flex: 1;
    min-width: 0;
    position: relative;
  }

  .doc-name {
    font-weight: 500;
    font-size: 14px;
    color: #1f2937;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .doc-id {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
  }

  .primary-tag {
    position: absolute;
    top: -8px;
    right: 0;
    background-color: #4f46e5;
    color: white;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .checkbox-wrapper {
    margin-left: 12px;
  }

  .checkbox {
    width: 20px;
    height: 20px;
    border: 2px solid #d1d5db;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    background-color: white;
  }

  .checkbox.checked {
    background-color: #4f46e5;
    border-color: #4f46e5;
  }

  .modal-footer {
    padding: 16px 20px;
    border-top: 1px solid #e5e7eb;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #f9fafb;
  }

  .selection-summary {
    font-size: 14px;
    color: #4b5563;
  }

  .selection-max {
    font-size: 12px;
    color: #6366f1;
    margin-left: 4px;
  }

  .button-group {
    display: flex;
    gap: 12px;
  }

  .cancel-button {
    padding: 8px 16px;
    background-color: white;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    color: #4b5563;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .cancel-button:hover {
    background-color: #f9fafb;
  }

  .activate-button {
    padding: 8px 16px;
    background-color: #4a63ee;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .activate-button:hover:not(:disabled) {
    background-color: #3a53de;
  }

  .activate-button:disabled {
    background-color: #9ca3af;
    cursor: not-allowed;
  }
</style>

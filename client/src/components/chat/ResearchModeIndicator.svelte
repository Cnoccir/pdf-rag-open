<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import Icon from '$c/Icon.svelte';

  // Add default values to all props
  export let activeDocuments = [];
  export let recommendedDocuments = [];
  export let primaryDocumentId = null; // ID of primary document

  const dispatch = createEventDispatcher();
  let expanded = false;
  let mounted = false;

  // Create safe versions of all props
  $: safeActiveDocuments = Array.isArray(activeDocuments) ? activeDocuments : [];
  $: safeRecommendedDocuments = Array.isArray(recommendedDocuments) ? recommendedDocuments : [];
  $: safePrimaryDocumentId = primaryDocumentId || "";

  // Debug logging
  $: console.log("Active documents in indicator:", safeActiveDocuments);
  $: console.log("Primary document ID:", safePrimaryDocumentId);
  $: console.log("Recommended documents:", safeRecommendedDocuments);

  // Force component refresh when data changes
  $: if (mounted && (safeActiveDocuments.length || safeRecommendedDocuments.length)) {
    console.log("Research data updated - refreshing component");
    checkDocumentData();
  }

  onMount(() => {
    mounted = true;
    checkDocumentData();
  });

  function checkDocumentData() {
    // Verify document data integrity
    if (safeActiveDocuments.length > 0) {
      const hasPrimary = safeActiveDocuments.some(doc => isPrimaryDocument(doc));
      if (!hasPrimary && safePrimaryDocumentId) {
        console.warn("Primary document not found in active documents - data inconsistency");
      }
    }
  }

  function toggleExpanded() {
    expanded = !expanded;
  }

  function handleAddDocument(documentId) {
    console.log("Adding document:", documentId);
    if (documentId) {
      dispatch('addDocument', { documentId });
    }
  }

  function handleRemoveDocument(documentId) {
    console.log("Removing document:", documentId);
    // Prevent removing the primary document
    if (documentId && !isPrimaryDocument({ id: documentId })) {
      dispatch('removeDocument', { documentId });
    } else {
      console.warn("Attempted to remove primary document, which is not allowed");
    }
  }

  function handleOpenDocument(doc) {
    console.log("Opening document:", doc?.id);
    if (doc && doc.id) {
      dispatch('openDocument', { documentId: doc.id, name: getDocumentName(doc) });
    }
  }

  function isPrimaryDocument(document) {
    // Safe check with null handling
    if (!document || !document.id || !safePrimaryDocumentId) return false;

    // Ensure the ID comparison is done correctly with string conversion
    return String(document.id) === String(safePrimaryDocumentId);
  }

  function getDocumentName(doc) {
    if (!doc) return "Unknown Document";

    // First try to use the document's name property
    if (doc.name && doc.name !== `Document ${doc.id}`) {
      return doc.name;
    }

    // If the document is the primary document, try to use its name from the conversation
    if (isPrimaryDocument(doc)) {
      // Try to find the primary document's name from active documents
      const primaryDoc = safeActiveDocuments.find(d => isPrimaryDocument(d));
      if (primaryDoc && primaryDoc.name) {
        return primaryDoc.name;
      }
    }

    // Fall back to using just the ID
    return `Document ${doc.id}`;
  }

  function showDocumentSelector() {
    dispatch('showDocumentSelector');
  }

  // Handle errors if data is inconsistent
  function getActiveDocumentsSafe() {
    // Ensures we always have an array even if data is incorrect
    if (!safeActiveDocuments || !Array.isArray(safeActiveDocuments)) {
      console.error("Active documents is not an array:", safeActiveDocuments);
      return [];
    }
    return safeActiveDocuments;
  }

  function getRecommendedDocumentsSafe() {
    // Ensures we always have an array even if data is incorrect
    if (!safeRecommendedDocuments || !Array.isArray(safeRecommendedDocuments)) {
      console.error("Recommended documents is not an array:", safeRecommendedDocuments);
      return [];
    }
    return safeRecommendedDocuments;
  }
</script>

<div class="research-mode-indicator bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-3 mb-4">
  <div class="header flex justify-between items-center">
    <div class="flex items-center gap-2">
      <div class="bg-blue-600 text-white p-1.5 rounded-md">
        <Icon name="search" size="16px" />
      </div>
      <h3 class="font-medium text-blue-800">Research Mode Active</h3>
    </div>

    <div class="flex items-center gap-2">
      <button
        class="text-blue-600 text-sm hover:text-blue-800 transition-colors"
        on:click={toggleExpanded}
      >
        {expanded ? 'Hide Details' : 'Show Details'}
      </button>

      <button
        class="text-blue-600 text-sm hover:text-blue-800 transition-colors border border-blue-300 px-2 py-1 rounded"
        on:click={showDocumentSelector}
      >
        Edit Documents
      </button>
    </div>
  </div>

  {#if expanded}
    <div class="mt-3 space-y-3">
      <div>
        <h4 class="text-sm font-medium text-blue-700 mb-2">Active Documents</h4>
        <div class="space-y-1.5">
          {#each getActiveDocumentsSafe() as doc}
            <div class="document-item flex items-center justify-between bg-white p-2 rounded border border-blue-100">
              <div class="flex items-center flex-1 min-w-0">
                <Icon name="description" outlined size="14px" class="text-blue-600 mr-2 flex-shrink-0" />
                <span class="text-sm truncate">{getDocumentName(doc)}</span>
                {#if isPrimaryDocument(doc)}
                  <span class="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full font-bold flex-shrink-0">Primary</span>
                {/if}
              </div>

              <div class="flex items-center gap-1 ml-2 flex-shrink-0">
                <!-- View document button -->
                <button
                  class="view-document-btn"
                  on:click|stopPropagation={() => handleOpenDocument(doc)}
                  title="View document"
                >
                  <Icon name="visibility" size="14px" />
                </button>

                {#if !isPrimaryDocument(doc)}
                  <!-- Only show remove button for non-primary documents -->
                  <button
                    class="remove-document-btn"
                    on:click|stopPropagation={() => handleRemoveDocument(doc.id)}
                    title="Remove from research"
                  >
                    <Icon name="close" size="14px" />
                  </button>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>

      {#if getRecommendedDocumentsSafe().length > 0}
        <div>
          <h4 class="text-sm font-medium text-blue-700 mb-2">Recommended Documents</h4>
          <div class="space-y-2">
            {#each getRecommendedDocumentsSafe() as doc}
              <div class="flex justify-between items-center bg-white p-2 rounded border border-blue-100">
                <div class="flex-1 min-w-0">
                  <div class="text-sm font-medium truncate">{getDocumentName(doc)}</div>
                  {#if doc.concepts && doc.concepts.length > 0}
                    <div class="text-xs text-blue-600 mt-1 truncate">
                      <span class="font-medium">Concepts:</span> {doc.concepts.slice(0, 3).join(', ')}
                    </div>
                  {/if}
                </div>
                <div class="flex items-center gap-1 ml-2 flex-shrink-0">
                  <button
                    class="view-document-btn"
                    on:click|stopPropagation={() => handleOpenDocument(doc)}
                    title="View document"
                  >
                    <Icon name="visibility" size="14px" />
                  </button>

                  <button
                    class="add-document-btn bg-blue-600 text-white text-xs px-3 py-1 rounded hover:bg-blue-700 transition-colors"
                    on:click|stopPropagation={() => handleAddDocument(doc.id)}
                    title="Add to research"
                  >
                    Add
                  </button>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .research-mode-indicator {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }

  h3 {
    font-size: 15px;
  }

  h4 {
    font-size: 14px;
    display: flex;
    align-items: center;
  }

  h4::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background-color: currentColor;
    margin-right: 6px;
  }

  .view-document-btn,
  .remove-document-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    border: none;
    background-color: transparent;
    transition: all 0.2s;
  }

  .view-document-btn {
    color: #3b82f6;
  }

  .view-document-btn:hover {
    background-color: #e0f2fe;
  }

  .remove-document-btn {
    color: #ef4444;
  }

  .remove-document-btn:hover {
    background-color: #fee2e2;
  }

  .document-item {
    transition: all 0.2s;
  }

  .document-item:hover {
    border-color: #93c5fd;
    background-color: #f0f9ff;
  }
</style>

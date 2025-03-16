{#await loadThumbnails()}
  <div class="loading-container">
    <div class="loading-spinner"></div>
  </div>
{:then}
  <div class="document-controls">
    <div class="view-controls">
      <button class:active={viewMode === 'grid'} on:click={() => viewMode = 'grid'}>
        <i class="fas fa-th-large"></i>
      </button>
      <button class:active={viewMode === 'list'} on:click={() => viewMode = 'list'}>
        <i class="fas fa-list"></i>
      </button>
    </div>
    <div class="sort-controls">
      <select bind:value={sortBy}>
        <option value="name">Name</option>
        <option value="date">Date</option>
        <option value="size">Size</option>
      </select>
      <button on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}>
        <i class="fas fa-sort-{sortOrder === 'asc' ? 'up' : 'down'}"></i>
      </button>
    </div>
    <div class="bulk-controls" class:visible={selectedDocs.length > 0}>
      <button on:click={handleBulkDelete}>Delete Selected</button>
      <button on:click={handleBulkTag}>Add Tags</button>
      <button on:click={handleBulkMove}>Move to Folder</button>
    </div>
  </div>

  <div class="documents-container" class:grid-view={viewMode === 'grid'} class:list-view={viewMode === 'list'}>
    {#each sortedDocuments as doc (doc.id)}
      <div
        class="document-item"
        class:selected={selectedDocs.includes(doc.id)}
        on:mouseenter={() => handleDocHover(doc)}
        on:mouseleave={handleDocLeave}
      >
        <div class="select-box">
          <input
            type="checkbox"
            checked={selectedDocs.includes(doc.id)}
            on:change={(e) => handleDocSelect(doc.id, e.target.checked)}
          />
        </div>

        <!-- Fix A11y warning by using button with keyboard events -->
        <button
          class="thumbnail-button"
          on:click={() => handleDocClick(doc)}
          on:keydown={(e) => e.key === 'Enter' && handleDocClick(doc)}
          aria-label={`Open ${doc.name}`}
        >
          <div class="thumbnail">
            {#if doc.thumbnail}
              <img src={doc.thumbnail} alt={doc.name} />
            {:else}
              <div class="placeholder">
                <i class="far fa-file-pdf"></i>
              </div>
            {/if}
          </div>
        </button>

        <!-- Fix A11y warning by using button with keyboard events -->
        <button
          class="doc-info-button"
          on:click={() => handleDocClick(doc)}
          on:keydown={(e) => e.key === 'Enter' && handleDocClick(doc)}
          aria-label={`View details for ${doc.name}`}
        >
          <div class="doc-info">
            <h3>{doc.name}</h3>
            <div class="metadata">
              <span>{formatDate(doc.date)}</span>
              <span>{formatSize(doc.size)}</span>
            </div>
            <div class="tags">
              {#each doc.tags || [] as tag}
                <span class="tag">{tag}</span>
              {/each}
            </div>

            <!-- Add document summary information if available -->
            {#if doc.metadata && doc.metadata.document_summary && showSummaries}
              <div class="doc-summary">
                {#if doc.metadata.document_summary.primary_concepts?.length > 0}
                  <div class="summary-concepts">
                    {#each doc.metadata.document_summary.primary_concepts.slice(0, 3) as concept}
                      <span class="concept-tag">{concept}</span>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}
          </div>
        </button>

        <div class="doc-actions">
          <button
            class="action-button view-button"
            on:click|stopPropagation={() => handleDocClick(doc)}
            title="View Document"
            aria-label="View document"
          >
            <i class="fas fa-eye"></i>
          </button>

          <button
            class="action-button retrigger-button"
            on:click|stopPropagation={() => handleReTriggerDoc(doc)}
            title="Re-trigger Processing"
            aria-label="Re-process document"
          >
            <i class="fas fa-sync-alt"></i>
          </button>

          <button
            class="action-button delete-button"
            on:click|stopPropagation={() => handleDeleteDoc(doc)}
            title="Delete Document"
            aria-label="Delete document"
          >
            <i class="fas fa-trash-alt"></i>
          </button>
        </div>

        {#if hoveredDoc?.id === doc.id}
          <div class="preview-popup">
            <div class="preview-thumbnail">
              <img src={doc.thumbnail} alt={doc.name} />
            </div>
            <div class="preview-details">
              <h4>{doc.name}</h4>
              <p>Created: {formatDate(doc.date)}</p>
              <p>Size: {formatSize(doc.size)}</p>
              {#if doc.tags?.length > 0}
                <p>Tags: {doc.tags.join(', ')}</p>
              {/if}

              <!-- Add document summary preview -->
              {#if doc.metadata?.document_summary}
                <div class="preview-summary">
                  <h5>Summary</h5>
                  {#if doc.metadata.document_summary.primary_concepts?.length > 0}
                    <p>Key Concepts: {doc.metadata.document_summary.primary_concepts.slice(0, 3).join(', ')}</p>
                  {/if}
                  {#if doc.metadata.document_summary.key_insights?.length > 0}
                    <p>Insight: {truncate(doc.metadata.document_summary.key_insights[0], 60)}</p>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/each}
  </div>
{/await}

<script>
  import { onMount } from 'svelte';
  import { pdfjsLib } from 'pdfjs-dist';
  import { formatDate, formatSize } from '../utils/formatters';
  import { deleteDocument, triggerEmbedding } from '$s/documents';

  export let documents = [];
  export let showSummaries = true; // Option to show summaries in the grid

  let viewMode = 'grid';
  let sortBy = 'date';
  let sortOrder = 'desc';
  let selectedDocs = [];
  let hoveredDoc = null;

  $: sortedDocuments = [...documents].sort((a, b) => {
    const modifier = sortOrder === 'asc' ? 1 : -1;
    switch (sortBy) {
      case 'name':
        return modifier * a.name.localeCompare(b.name);
      case 'date':
        if (!a.created_at || !b.created_at) return 0;
        return modifier * (new Date(b.created_at) - new Date(a.created_at));
      case 'size':
        return modifier * ((b.size || 0) - (a.size || 0));
      default:
        return 0;
    }
  });

  async function loadThumbnails() {
    for (const doc of documents) {
      if (!doc.thumbnail) {
        try {
          // Try to get thumbnail from server or generate on client
          try {
            // First try to get from server
            const response = await fetch(`/api/pdfs/${doc.id}/thumbnail`);
            if (response.ok) {
              const blob = await response.blob();
              doc.thumbnail = URL.createObjectURL(blob);
              continue;
            }
          } catch (error) {
            console.log('No server thumbnail, generating locally');
          }

          // If server thumbnail not available, generate locally
          const pdfUrl = `/api/pdfs/${doc.id}/content`;
          const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
          const page = await pdf.getPage(1);
          const viewport = page.getViewport({ scale: 0.2 });

          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.width = viewport.width;
          canvas.height = viewport.height;

          await page.render({
            canvasContext: context,
            viewport: viewport
          }).promise;

          doc.thumbnail = canvas.toDataURL();
        } catch (error) {
          console.error(`Error generating thumbnail for ${doc.name}:`, error);
        }
      }
    }
  }

  function handleDocClick(doc) {
    // Navigate to document viewer
    window.location.href = `/view/${doc.id}`;
  }

  function handleDocHover(doc) {
    hoveredDoc = doc;
  }

  function handleDocLeave() {
    hoveredDoc = null;
  }

  function handleDocSelect(docId, selected) {
    selectedDocs = selected
      ? [...selectedDocs, docId]
      : selectedDocs.filter(id => id !== docId);
  }

  // New function to handle document deletion
  async function handleDeleteDoc(doc) {
    if (confirm(`Are you sure you want to delete "${doc.name}"?`)) {
      try {
        const success = await deleteDocument(doc.id);
        if (success) {
          // Remove from the local documents array
          documents = documents.filter(d => d.id !== doc.id);
          // Remove from selected docs if it was selected
          selectedDocs = selectedDocs.filter(id => id !== doc.id);
        }
      } catch (error) {
        console.error('Failed to delete document:', error);
        alert(`Error: ${error.message || 'Failed to delete document'}`);
      }
    }
  }

  // New function to re-trigger document processing
  async function handleReTriggerDoc(doc) {
    try {
      const success = await triggerEmbedding(doc.id);
      if (success) {
        alert(`Processing re-triggered for ${doc.name}`);
      }
    } catch (error) {
      console.error('Failed to re-trigger processing:', error);
      alert(`Error: ${error.message || 'Failed to re-trigger processing'}`);
    }
  }

  // Add bulk delete function
  async function handleBulkDelete() {
    if (confirm(`Are you sure you want to delete ${selectedDocs.length} selected documents?`)) {
      let successCount = 0;

      for (const docId of selectedDocs) {
        try {
          const success = await deleteDocument(docId);
          if (success) {
            successCount++;
          }
        } catch (error) {
          console.error(`Error deleting document ${docId}:`, error);
        }
      }

      // Update the documents list
      documents = documents.filter(doc => !selectedDocs.includes(doc.id));
      // Clear selected documents
      selectedDocs = [];

      if (successCount > 0) {
        alert(`Successfully deleted ${successCount} documents`);
      }
    }
  }

  // Helper function to truncate text
  function truncate(text, maxLength = 60) {
    if (!text) return '';
    return text.length > maxLength
      ? text.substring(0, maxLength) + '...'
      : text;
  }

  function handleBulkTag() {
    // Implement tag dialog
    alert('Bulk tagging not implemented yet');
  }

  function handleBulkMove() {
    // Implement move dialog
    alert('Bulk move not implemented yet');
  }
</script>

<style>
  .documents-container {
    padding: 2rem;
    display: flex;
    gap: 1.5rem;
  }

  .grid-view {
    flex-wrap: wrap;
  }

  .list-view {
    flex-direction: column;
  }

  .document-controls {
    padding: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f5f5f5;
    border-bottom: 1px solid #e0e0e0;
  }

  .view-controls button,
  .sort-controls button {
    padding: 0.5rem;
    background: none;
    border: 1px solid #ddd;
    cursor: pointer;
  }

  .view-controls button.active {
    background: #e0e0e0;
  }

  .document-item {
    position: relative;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s ease;
  }

  .grid-view .document-item {
    width: calc(25% - 1.5rem);
  }

  .list-view .document-item {
    display: flex;
    align-items: center;
    padding: 1rem;
  }

  .document-item:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }

  /* Style buttons to look like divs */
  .thumbnail-button, .doc-info-button {
    background: none;
    border: none;
    padding: 0;
    margin: 0;
    text-align: left;
    width: 100%;
    cursor: pointer;
  }

  .thumbnail {
    aspect-ratio: 1;
    background: #f5f5f5;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .grid-view .thumbnail {
    width: 100%;
  }

  .list-view .thumbnail {
    width: 60px;
    height: 60px;
  }

  .thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .doc-info {
    padding: 1rem;
  }

  .metadata {
    font-size: 0.875rem;
    color: #666;
  }

  .tags {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }

  .tag {
    background: #e0e0e0;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
  }

  /* Add styles for summary concepts */
  .doc-summary {
    margin-top: 0.5rem;
    border-top: 1px solid #f0f0f0;
    padding-top: 0.5rem;
  }

  .summary-concepts {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    margin-top: 0.25rem;
  }

  .concept-tag {
    background-color: #e9f2fe;
    color: #3182ce;
    padding: 0.1rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 500;
  }

  .doc-actions {
    display: flex;
    flex-direction: row;
    gap: 0.25rem;
    padding: 0.5rem;
  }

  .list-view .doc-actions {
    flex-direction: row;
    align-items: center;
  }

  .action-button {
    background: none;
    border: none;
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
  }

  .action-button:hover {
    background-color: #f1f5f9;
  }

  .view-button:hover {
    color: #3182ce;
  }

  .retrigger-button:hover {
    color: #805ad5;
  }

  .delete-button:hover {
    color: #e53e3e;
  }

  .preview-popup {
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 1rem;
    z-index: 10;
    min-width: 300px;
    display: none;
  }

  .document-item:hover .preview-popup {
    display: block;
  }

  .preview-summary {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #f1f5f9;
  }

  .preview-summary h5 {
    margin: 0 0 4px 0;
    font-size: 14px;
  }

  .preview-summary p {
    margin: 4px 0;
    font-size: 12px;
  }

  .bulk-controls {
    display: none;
  }

  .bulk-controls.visible {
    display: flex;
    gap: 1rem;
  }

  .loading-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 200px;
  }

  .loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
</style>

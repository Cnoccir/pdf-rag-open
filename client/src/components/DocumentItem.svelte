<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { deleteDocument } from '$s/documents';
  import Icon from './Icon.svelte'; // Assuming you have an Icon component

  export let document;
  export let showSummary = false;

  const dispatch = createEventDispatcher();

  async function handleDelete() {
    if (confirm(`Are you sure you want to delete "${document.name}"?`)) {
      const success = await deleteDocument(document.id);
      if (success) {
        dispatch('deleted', { id: document.id });
      }
    }
  }

  function handleView() {
    dispatch('view', { id: document.id });
  }

  function handleReTrigger() {
    dispatch('retrigger', { id: document.id });
  }

  // Function to truncate summary text
  function truncate(text, maxLength = 100) {
    if (!text) return '';
    return text.length > maxLength
      ? text.substring(0, maxLength) + '...'
      : text;
  }
</script>

<div class="document-item">
  <div class="document-header">
    <h3 class="document-title">{document.name}</h3>

    <div class="document-actions">
      <button
        class="action-button view-button"
        on:click|stopPropagation={handleView}
        title="View Document"
      >
        <Icon name="visibility" size="18px" />
      </button>

      <button
        class="action-button retrigger-button"
        on:click|stopPropagation={handleReTrigger}
        title="Re-trigger Processing"
      >
        <Icon name="refresh" size="18px" />
      </button>

      <button
        class="action-button delete-button"
        on:click|stopPropagation={handleDelete}
        title="Delete Document"
      >
        <Icon name="delete" size="18px" />
      </button>
    </div>
  </div>

  {#if document.description}
    <p class="document-description">{truncate(document.description, 150)}</p>
  {/if}

  {#if showSummary && document.metadata && document.metadata.document_summary}
    <div class="document-summary">
      {#if document.metadata.document_summary.title}
        <p class="summary-title"><strong>Title:</strong> {document.metadata.document_summary.title}</p>
      {/if}

      {#if document.metadata.document_summary.primary_concepts && document.metadata.document_summary.primary_concepts.length > 0}
        <p class="summary-concepts">
          <strong>Key Concepts:</strong>
          {document.metadata.document_summary.primary_concepts.slice(0, 5).join(', ')}
        </p>
      {/if}

      {#if document.metadata.document_summary.key_insights && document.metadata.document_summary.key_insights.length > 0}
        <div class="summary-insights">
          <strong>Key Insights:</strong>
          <ul>
            {#each document.metadata.document_summary.key_insights.slice(0, 3) as insight}
              <li>{truncate(insight, 120)}</li>
            {/each}
          </ul>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .document-item {
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: white;
    transition: all 0.2s ease;
    margin-bottom: 1rem;
  }

  .document-item:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  }

  .document-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .document-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: #2d3748;
  }

  .document-actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-button {
    background: none;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    transition: background-color 0.2s;
  }

  .action-button:hover {
    background-color: #f7fafc;
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

  .document-description {
    color: #718096;
    margin: 0.5rem 0;
    font-size: 0.875rem;
  }

  .document-summary {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.875rem;
  }

  .summary-title, .summary-concepts {
    margin: 0.5rem 0;
  }

  .summary-insights ul {
    margin: 0.25rem 0;
    padding-left: 1.5rem;
  }

  .summary-insights li {
    margin-bottom: 0.25rem;
  }
</style>

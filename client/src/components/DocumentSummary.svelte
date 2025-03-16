<script lang="ts">
  export let summary = {};
  export let categoryInfo = {};
  export let truncateLength = 80;
  export let compact = false; // Option for a more compact display

  // Check if summary exists and has content
  $: hasSummary = summary && (
    (summary.primary_concepts && summary.primary_concepts.length) ||
    (summary.title) ||
    (summary.key_insights && summary.key_insights.length) ||
    (summary.section_structure && summary.section_structure.length)
  );

  function truncate(text, maxLength = truncateLength) {
    if (!text || typeof text !== 'string') return '';
    return text.length > maxLength
      ? text.substring(0, maxLength) + '...'
      : text;
  }

  // Safely get primary concepts
  function getSafePrimaryConcepts() {
    if (!summary) return [];
    if (!summary.primary_concepts) return [];

    // Handle both array and object formats
    if (Array.isArray(summary.primary_concepts)) {
      return summary.primary_concepts.slice(0, compact ? 3 : 5);
    } else if (typeof summary.primary_concepts === 'object') {
      return Object.keys(summary.primary_concepts).slice(0, compact ? 3 : 5);
    }
    return [];
  }

  // Get safe concepts
  $: primaryConcepts = getSafePrimaryConcepts();
</script>

<div class="document-summary {compact ? 'compact' : ''}">
  {#if !compact}
    <h3 class="summary-heading">Document Summary</h3>
  {/if}

  {#if hasSummary}
    {#if summary.title && !compact}
      <div class="summary-section">
        <h4>Title</h4>
        <p>{summary.title}</p>
      </div>
    {/if}

    {#if primaryConcepts && primaryConcepts.length > 0}
      <div class="summary-section">
        {#if !compact}<h4>Key Concepts</h4>{/if}
        <div class="concept-tags">
          {#each primaryConcepts as concept}
            <span class="concept-tag">{concept}</span>
          {/each}
        </div>
      </div>
    {/if}

    {#if summary.key_insights && summary.key_insights.length > 0 && !compact}
      <div class="summary-section">
        <h4>Key Insights</h4>
        <ul class="insights-list">
          {#each summary.key_insights.slice(0, 3) as insight}
            <li>{truncate(insight, truncateLength * 2)}</li>
          {/each}
        </ul>
      </div>
    {/if}

    {#if summary.section_structure && summary.section_structure.length > 0 && !compact}
      <div class="summary-section">
        <h4>Document Structure</h4>
        <ul class="structure-list">
          {#each summary.section_structure.slice(0, 5) as section}
            <li>{truncate(section)}</li>
          {/each}
        </ul>
      </div>
    {/if}

    {#if categoryInfo && Object.keys(categoryInfo).length > 0 && !compact}
      <div class="summary-section">
        <h4>Document Category</h4>
        <div class="category-info">
          <span class="category-tag">{categoryInfo.current_category || 'general'}</span>
          {#if categoryInfo.predicted_category && categoryInfo.predicted_category !== categoryInfo.current_category}
            <p class="predicted-category">
              Suggested: <span class="category-tag suggested">{categoryInfo.predicted_category}</span>
            </p>
          {/if}
        </div>
      </div>
    {/if}
  {:else}
    <p class="no-summary">No summary information available.</p>
  {/if}
</div>
<style>
  .document-summary {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
  }

  .document-summary.compact {
    background-color: transparent;
    border: none;
    padding: 0;
  }

  .summary-heading {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 1rem;
    color: #2d3748;
  }

  .summary-section {
    margin-bottom: 1.25rem;
  }

  .compact .summary-section {
    margin-bottom: 0.5rem;
  }

  .summary-section h4 {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 0.5rem;
    color: #4a5568;
  }

  .summary-section p {
    margin: 0.25rem 0;
    color: #4a5568;
    line-height: 1.5;
  }

  .concept-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .compact .concept-tags {
    gap: 0.25rem;
  }

  .concept-tag {
    background-color: #e9f2fe;
    color: #3182ce;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .compact .concept-tag {
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
  }

  .insights-list, .structure-list {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
  }

  .insights-list li, .structure-list li {
    margin-bottom: 0.5rem;
    color: #4a5568;
  }

  .category-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .category-tag {
    display: inline-block;
    background-color: #e9f2fe;
    color: #3182ce;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
    text-transform: capitalize;
  }

  .category-tag.suggested {
    background-color: #fed7e2;
    color: #d53f8c;
  }

  .predicted-category {
    font-size: 0.875rem;
    margin: 0;
  }

  .no-summary {
    color: #718096;
    font-style: italic;
  }

  .compact .no-summary {
    margin: 0;
    font-size: 0.75rem;
  }
</style>

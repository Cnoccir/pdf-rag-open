<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import DocumentGrid from '../components/DocumentGrid.svelte';

	onMount(() => {
		goto('/documents');
	});
</script>

<div class="container">
  <header>
    <h1>Your Documents</h1>
    <div class="actions">
      <button class="new-doc">
        <i class="fas fa-plus"></i>
        New Document
      </button>
    </div>
  </header>

  <DocumentGrid />
</div>

<style>
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #333;
  }

  .actions {
    display: flex;
    gap: 1rem;
  }

  .new-doc {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: #1a73e8;
    color: white;
    border: none;
    border-radius: 4px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  .new-doc:hover {
    background: #1557b0;
  }
</style>

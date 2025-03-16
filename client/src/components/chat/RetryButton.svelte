<!-- RetryButton.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Icon from '$c/Icon.svelte';

  export let isLoading = false;

  const dispatch = createEventDispatcher();

  function handleClick() {
    if (!isLoading) {
      dispatch('click');
    }
  }
</script>

<button
  class="retry-button {isLoading ? 'loading' : ''}"
  on:click={handleClick}
  disabled={isLoading}
  aria-label="Regenerate response"
>
  <div class="button-content">
    <div class="icon-container">
      {#if isLoading}
        <div class="spinner"></div>
      {:else}
        <Icon name="refresh" size="16px" />
      {/if}
    </div>
    <span>Regenerate response</span>
  </div>
</button>

<style>
  .retry-button {
    width: 100%;
    max-width: 240px;
    padding: 8px 12px;
    background-color: #f0f4f8;
    border: 1px solid #d1dae6;
    border-radius: 6px;
    color: #4b5563;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }

  .retry-button:hover:not(.loading) {
    background-color: #e5eaf1;
    border-color: #b5c3d8;
    color: #374151;
  }

  .retry-button.loading {
    background-color: #f0f4f8;
    pointer-events: none;
    cursor: default;
  }

  .button-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .icon-container {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(75, 85, 99, 0.2);
    border-radius: 50%;
    border-top-color: #4b5563;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>

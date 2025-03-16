<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Conversation } from '$s/chat';
  import { api } from '$api';
  import Icon from '../Icon.svelte';
  import { store, setActiveConversationId } from '$s/chat';

  export let conversations: Conversation[] = [];
  const dispatch = createEventDispatcher();

  async function handleDelete(conversation: Conversation) {
    if (confirm('Are you sure you want to delete this conversation?')) {
      try {
        await api.delete(`/conversations/${conversation.id}`);
        dispatch('delete', { id: conversation.id });
      } catch (error) {
        console.error('Failed to delete conversation:', error);
      }
    }
  }

  function formatDate(dateStr: string) {
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      hour12: true
    }).format(date);
  }

  $: activeConversationId = $store.activeConversationId;
</script>

<div class="conversation-list">
  {#each conversations as conversation (conversation.id)}
    <div 
      class="conversation-item {conversation.id === activeConversationId ? 'active' : ''}"
      on:click={() => setActiveConversationId(conversation.id)}
    >
      <div class="conversation-content">
        <div class="conversation-title">
          {conversation.title || 'Untitled Conversation'}
        </div>
        <div class="conversation-date">
          {formatDate(conversation.last_updated)}
        </div>
      </div>
      <button
        class="delete-button"
        on:click|stopPropagation={() => handleDelete(conversation)}
        title="Delete conversation"
      >
        <Icon name="trash" size="16" />
      </button>
    </div>
  {/each}
</div>

<style>
  .conversation-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
  }

  .conversation-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px;
    border-radius: 8px;
    background: white;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid #e5e7eb;
  }

  .conversation-item:hover {
    background: #f3f4f6;
  }

  .conversation-item.active {
    background: #e5e7eb;
  }

  .conversation-content {
    flex: 1;
    min-width: 0;
  }

  .conversation-title {
    font-size: 14px;
    font-weight: 500;
    color: #111827;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .conversation-date {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
  }

  .delete-button {
    opacity: 0;
    padding: 6px;
    border-radius: 6px;
    color: #6b7280;
    transition: all 0.2s ease;
  }

  .conversation-item:hover .delete-button {
    opacity: 1;
  }

  .delete-button:hover {
    background: #fee2e2;
    color: #dc2626;
  }
</style>

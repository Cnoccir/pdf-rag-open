<!-- UserMessage.svelte -->
<script lang="ts">
  import { marked } from 'marked';

  export let content = '';
  export let timestamp = new Date();

  function formatTime(date: Date): string {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
</script>

<div class="user-message">
  <div class="message-content">
    <div class="message-timestamp">{formatTime(timestamp)}</div>
    {@html marked(content, { breaks: true, gfm: true })}
  </div>
  <div class="avatar">
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
</div>

<style>
  .user-message {
    display: flex;
    align-items: flex-start;
    margin: 16px 0 24px 0;
    gap: 12px;
    justify-content: flex-end;
  }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #64748b;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .message-content {
    background-color: #eef2ff;
    border-radius: 12px 12px 0 12px;
    padding: 16px;
    max-width: 75%;
    color: #1e293b;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    overflow-x: auto;
    animation: fadeIn 0.3s ease-out;
    position: relative;
  }

  .message-timestamp {
    position: absolute;
    top: 4px;
    right: 8px;
    font-size: 11px;
    color: #6b7280;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message-content :global(p) {
    margin: 0.5em 0;
  }

  .message-content :global(p:first-child) {
    margin-top: 0;
  }

  .message-content :global(p:last-child) {
    margin-bottom: 0;
  }

  .message-content :global(pre) {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 0.8em;
    border-radius: 4px;
    overflow-x: auto;
    margin: 0.8em 0;
    border: 1px solid #d1d5db;
  }

  .message-content :global(code) {
    font-family: 'Fira Code', 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
  }
</style>

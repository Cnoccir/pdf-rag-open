<!-- AssistantMessage.svelte -->
<script lang="ts">
  import { marked } from 'marked';
  import { scoreConversation } from '$s/chat';
  import Icon from '$c/Icon.svelte';
  import hljs from 'highlight.js';
  import DOMPurify from 'dompurify';

  export let content = '';
  export let timestamp = new Date();
  let score = 0;

  marked.setOptions({
    highlight: function(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : 'plaintext';
      return hljs.highlight(code, { language }).value;
    },
    langPrefix: 'hljs language-',
    breaks: true,
    gfm: true
  });

  function getSanitizedMarkdown(content: string) {
    const rawHtml = marked(content);
    return DOMPurify.sanitize(rawHtml);
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  async function applyScore(_score: number) {
    if (score !== 0) {
      return;
    }
    score = _score;
    return scoreConversation(_score);
  }
</script>

<div class="assistant-message">
  <div class="avatar">
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M9 11.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z" fill="currentColor" />
      <path d="M15.5 6.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" fill="currentColor" />
      <path d="M15.5 15.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" fill="currentColor" />
      <path d="M12 11a1 1 0 1 0 0 2h4a1 1 0 1 0 0-2h-4z" fill="currentColor" />
      <path d="M10.25 14.5a1 1 0 0 1 1 1c0 .28-.22.8-.78 1.22-.56.42-1.36.78-2.47.78-1.83 0-2.64-.8-2.9-1.94l-.02-.06 1.92-.5.02.06c.13.45.36.44 1.03.44.38 0 .7-.12.9-.25.19-.13.24-.23.25-.25a1 1 0 0 1 1.05-1z" fill="currentColor" />
      <path d="M13 5a1 1 0 0 1 1-1c.73 0 1.17.18 1.43.44.26.25.33.52.38.76l.02.07-1.9.59-.01-.05s-.08-.08-.16-.12c-.08-.04-.16-.04-.21-.01l-.05.03V5z" fill="currentColor" />
      <path d="M18 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0z" stroke="currentColor" stroke-width="2" />
    </svg>
  </div>
  <div class="message-content">
    <div class="message-timestamp">{formatTime(timestamp)}</div>
    {@html getSanitizedMarkdown(content)}
    <div class="score-buttons">
      <button
        class="score-button"
        class:active={score === 1}
        on:click={() => applyScore(1)}
        aria-label="Thumbs up"
      >
        <Icon name="thumb_up" outlined />
      </button>
      <button
        class="score-button"
        class:active={score === -1}
        on:click={() => applyScore(-1)}
        aria-label="Thumbs down"
      >
        <Icon name="thumb_down" outlined />
      </button>
    </div>
  </div>
</div>

<style>
  .assistant-message {
    display: flex;
    align-items: flex-start;
    margin: 16px 0 24px 0;
    gap: 12px;
    position: relative;
  }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background-color: #4a63ee;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .message-content {
    background-color: white;
    border-radius: 12px 12px 12px 0;
    padding: 16px;
    padding-top: 20px; /* Space for timestamp */
    max-width: 85%;
    overflow-x: auto;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    font-size: 15px;
    line-height: 1.6;
    color: #374151;
    animation: fadeIn 0.3s ease-out;
    position: relative;
  }

  .message-timestamp {
    position: absolute;
    top: 4px;
    right: 8px;
    font-size: 11px;
    color: #9ca3af;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Enhanced Headers */
  .message-content :global(h1) { font-size: 1.5em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 700; color: #1a202c; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.3em; }
  .message-content :global(h2) { font-size: 1.3em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 700; color: #1a202c; }
  .message-content :global(h3) { font-size: 1.1em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 600; color: #1a202c; }
  .message-content :global(h4) { font-size: 1em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 600; color: #1a202c; }
  .message-content :global(h5) { font-size: 0.9em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 600; color: #1a202c; }
  .message-content :global(h6) { font-size: 0.85em; margin-top: 1.2em; margin-bottom: 0.6em; font-weight: 600; color: #1a202c; }

  /* Basic Text Elements */
  .message-content :global(p) {
    margin: 0.8em 0;
  }

  .message-content :global(ul), .message-content :global(ol) {
    margin: 0.8em 0;
    padding-left: 2em;
  }

  .message-content :global(ol) {
    list-style-type: decimal;
  }

  .message-content :global(li) {
    margin: 0.4em 0;
  }

  .message-content :global(li::marker) {
    color: #4a63ee;
  }

  /* Enhanced Code Blocks */
  .message-content :global(pre) {
    background-color: #1e1e1e;
    padding: 1em;
    border-radius: 8px;
    overflow-x: auto;
    margin: 1em 0;
    border: 1px solid #2d2d2d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  }

  .message-content :global(pre code) {
    font-family: 'Fira Code', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    color: #d4d4d4;
    background-color: transparent;
    padding: 0;
    font-size: 0.9em;
    line-height: 1.5;
    white-space: pre;
  }

  /* Inline Code */
  .message-content :global(:not(pre) > code) {
    font-family: 'Fira Code', 'Monaco', 'Menlo', monospace;
    background-color: #f1f5f9;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
    color: #3b82f6;
    border: 1px solid #e2e8f0;
  }

  /* Scoring Buttons */
  .score-buttons {
    display: flex;
    gap: 8px;
    position: absolute;
    bottom: -32px;
    right: 10px;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .message-content:hover .score-buttons {
    opacity: 1;
  }

  .score-button {
    background: white;
    border: none;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.2s, transform 0.2s;
    padding: 6px;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .score-button:hover, .score-button.active {
    opacity: 1;
    transform: scale(1.1);
  }

  .score-button.active {
    color: #4a63ee;
  }
  /* Table styling */
  .message-content :global(table) {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0;
    font-size: 0.9em;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .message-content :global(thead) {
    background-color: #4a63ee;
    color: white;
    text-align: left;
    font-weight: bold;
  }

  .message-content :global(th),
  .message-content :global(td) {
    padding: 10px 12px;
    border: 1px solid #e2e8f0;
  }

  .message-content :global(th) {
    padding-top: 12px;
    padding-bottom: 12px;
  }

  .message-content :global(tr) {
    background-color: white;
  }

  .message-content :global(tr:nth-child(even)) {
    background-color: #f8fafc;
  }

  .message-content :global(tr:hover) {
    background-color: #f1f5f9;
  }

  /* Special styling for Yes/No values */
  .message-content :global(td:contains("Yes")) {
    color: #059669;
    font-weight: 600;
  }

  .message-content :global(td:contains("No")) {
    color: #dc2626;
  }

  /* Caption styling if present */
  .message-content :global(table caption) {
    margin-bottom: 8px;
    font-weight: 600;
    font-size: 0.95em;
    color: #4b5563;
  }  
</style>

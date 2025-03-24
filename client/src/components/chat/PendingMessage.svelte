<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import { store } from '$s/chat/store';

  // Use stream progress from store
  $: progress = $store.streamProgress;

  // Backup messages in case stream progress is not available
  const fallbackMessages = [
    "Searching document knowledge...",
    "Analyzing relevant content...",
    "Connecting concepts...",
    "Synthesizing information...",
    "Generating response..."
  ];

  let currentMessageIndex = 0;
  let displayedMessage = '';
  let intervalId: ReturnType<typeof setInterval>;
  let fallbackProgress = 0;
  let finishedInitialMessages = false;

  // Stage to icon mapping
  const stageIcons = {
    "initialization": "settings",
    "preparation": "library_books",
    "connection": "cloud_sync",
    "database_ready": "database",
    "query_analyzer": "search",
    "retriever": "find_in_page",
    "research_synthesizer": "compare",
    "knowledge_generator": "psychology",
    "response_generator": "rate_review",
    "finalizing": "check_circle",
    "complete": "done_all",
    "error": "error"
  };

  function typewriterEffect(message: string, index = 0) {
    if (index < message.length) {
      displayedMessage = message.slice(0, index + 1);
      setTimeout(() => typewriterEffect(message, index + 1), 25);
    }
  }

  function nextMessage() {
    if (currentMessageIndex < fallbackMessages.length - 1) {
      currentMessageIndex += 1;
      typewriterEffect(fallbackMessages[currentMessageIndex]);
      // Update progress based on pipeline stage
      fallbackProgress = (currentMessageIndex + 1) * (100 / fallbackMessages.length);
    } else {
      finishedInitialMessages = true;
      displayedMessage = "Finalizing response...";
    }
  }

  function startProgress() {
    if (progress.active) {
      // Use stream progress if available
      displayedMessage = progress.message;
    } else {
      // Fallback to basic messaging
      typewriterEffect(fallbackMessages[currentMessageIndex]);
      intervalId = setInterval(() => {
        if (!finishedInitialMessages) {
          nextMessage();
        }
      }, 2200); // Show each stage for 2.2 seconds
    }
  }

  onMount(() => {
    startProgress();
  });

  onDestroy(() => {
    if (intervalId) clearInterval(intervalId);
  });

  // Get actual progress percentage to display
  $: displayProgress = progress.active ? progress.percentage : fallbackProgress;

  // Get appropriate icon for current stage
  $: stageIcon = progress.active && progress.stage ? stageIcons[progress.stage] || "sync" : "sync";
</script>

<div class="pending-message" transition:fade={{ duration: 300 }}>
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
    <div class="process-info">
      {#if progress.active}
        <div class="stage-indicator">
          <span class="stage-icon material-icons">{stageIcon}</span>
          <span class="stage-badge">{progress.stage || "Processing"}</span>
        </div>
      {/if}
      <div class="message">{progress.active ? progress.message : displayedMessage}</div>
    </div>

    <!-- Processing stages visualization -->
    <div class="stages-container">
      <div class="stages">
        <div class="stage {progress.stage === 'query_analyzer' ? 'active' : (displayProgress > 25 ? 'completed' : '')}">
          <span class="stage-dot"></span>
          <span class="stage-label">Analyzing</span>
        </div>
        <div class="stage {progress.stage === 'retriever' ? 'active' : (displayProgress > 50 ? 'completed' : '')}">
          <span class="stage-dot"></span>
          <span class="stage-label">Searching</span>
        </div>
        <div class="stage {['knowledge_generator', 'research_synthesizer'].includes(progress.stage) ? 'active' : (displayProgress > 70 ? 'completed' : '')}">
          <span class="stage-dot"></span>
          <span class="stage-label">Processing</span>
        </div>
        <div class="stage {progress.stage === 'response_generator' ? 'active' : (displayProgress > 85 ? 'completed' : '')}">
          <span class="stage-dot"></span>
          <span class="stage-label">Responding</span>
        </div>
        <div class="stage {progress.stage === 'finalizing' || progress.stage === 'complete' ? 'active' : (displayProgress >= 100 ? 'completed' : '')}">
          <span class="stage-dot"></span>
          <span class="stage-label">Finishing</span>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .pending-message {
    display: flex;
    align-items: flex-start;
    margin: 16px 0 24px 0;
    gap: 12px;
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
    min-width: 280px;
    max-width: 85%;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    animation: pulse 2s infinite ease-in-out;
  }

  @keyframes pulse {
    0% { box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); }
    50% { box-shadow: 0 2px 12px rgba(74, 99, 238, 0.2); }
    100% { box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); }
  }

  .process-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 12px;
  }

  .stage-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .stage-icon {
    font-size: 16px;
    color: #4a63ee;
  }

  .stage-badge {
    display: inline-block;
    padding: 3px 8px;
    background-color: #e9f2ff;
    color: #4a63ee;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    align-self: flex-start;
    margin-bottom: 4px;
    text-transform: capitalize;
  }

  .message {
    font-size: 15px;
    color: #4b5563;
    font-weight: 500;
  }

  /* New stages visualization */
  .stages-container {
    margin: 16px 0;
  }

  .stages {
    display: flex;
    justify-content: space-between;
    position: relative;
    margin-bottom: 8px;
  }

  .stages:before {
    content: '';
    position: absolute;
    top: 8px;
    left: 10px;
    right: 10px;
    height: 2px;
    background-color: #e2e8f0;
    z-index: 1;
  }

  .stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 2;
  }

  .stage-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background-color: #e2e8f0;
    border: 2px solid #f8fafc;
    margin-bottom: 4px;
  }

  .stage.active .stage-dot {
    background-color: #4a63ee;
    box-shadow: 0 0 0 3px rgba(74, 99, 238, 0.2);
    animation: pulse-dot 1.5s infinite;
  }

  .stage.completed .stage-dot {
    background-color: #10b981;
  }

  @keyframes pulse-dot {
    0% { box-shadow: 0 0 0 0 rgba(74, 99, 238, 0.5); }
    70% { box-shadow: 0 0 0 6px rgba(74, 99, 238, 0); }
    100% { box-shadow: 0 0 0 0 rgba(74, 99, 238, 0); }
  }

  .stage-label {
    font-size: 10px;
    font-weight: 500;
    color: #6b7280;
  }

  .stage.active .stage-label {
    color: #4a63ee;
    font-weight: 600;
  }

  .stage.completed .stage-label {
    color: #10b981;
  }

</style>

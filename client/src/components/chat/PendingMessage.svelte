<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';

  const messages = [
    "Searching document knowledge...",
    "Analyzing relevant content...",
    "Connecting concepts...",
    "Synthesizing information...",
    "Generating response..."
  ];

  let currentMessageIndex = 0;
  let displayedMessage = '';
  let intervalId: ReturnType<typeof setInterval>;
  let progress = 0;
  let finishedInitialMessages = false;

  function typewriterEffect(message: string, index = 0) {
    if (index < message.length) {
      displayedMessage = message.slice(0, index + 1);
      setTimeout(() => typewriterEffect(message, index + 1), 25);
    }
  }

  function nextMessage() {
    if (currentMessageIndex < messages.length - 1) {
      currentMessageIndex += 1;
      typewriterEffect(messages[currentMessageIndex]);
      // Update progress based on pipeline stage
      progress = (currentMessageIndex + 1) * (100 / messages.length);
    } else {
      finishedInitialMessages = true;
      displayedMessage = "Finalizing response...";
    }
  }

  function startProgress() {
    typewriterEffect(messages[currentMessageIndex]);
    intervalId = setInterval(() => {
      if (!finishedInitialMessages) {
        nextMessage();
      }
    }, 2200); // Show each stage for 2.2 seconds
  }

  onMount(() => {
    startProgress();
  });

  onDestroy(() => {
    clearInterval(intervalId);
  });
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
    <div class="message">{displayedMessage}</div>
    <div class="progress-container">
      <div class="progress-bar">
        <div class="progress" style="width: {progress}%"></div>
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
    min-width: 260px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    animation: pulse 2s infinite ease-in-out;
  }

  @keyframes pulse {
    0% { box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); }
    50% { box-shadow: 0 2px 12px rgba(74, 99, 238, 0.2); }
    100% { box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); }
  }

  .message {
    font-size: 15px;
    color: #4b5563;
    margin-bottom: 12px;
    min-height: 20px;
    font-weight: 500;
  }

  .progress-container {
    margin-top: 8px;
  }

  .progress-bar {
    background-color: #e2e8f0;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
  }

  .progress {
    background-color: #4a63ee;
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease-in-out;
  }
</style>

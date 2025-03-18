<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '$api';
  import Icon from '$c/Icon.svelte';

  // State variables
  let systemHealth = null;
  let databaseHealth = null;
  let vectorStoreHealth = null;
  let memoryHealth = null;
  let loading = true;
  let error = null;
  let refreshInterval = null;
  let autoRefresh = false;
  let lastUpdated = '';

  // Fetch all health data
  async function fetchHealthData() {
    loading = true;
    error = null;

    try {
      // Fetch data in parallel
      const [systemRes, dbRes, vectorRes, memoryRes] = await Promise.all([
        api.get('/health/system'),
        api.get('/health/database'),
        api.get('/health/vector-store'),
        api.get('/health/memory')
      ]);

      systemHealth = systemRes.data;
      databaseHealth = dbRes.data;
      vectorStoreHealth = vectorRes.data;
      memoryHealth = memoryRes.data;

      lastUpdated = new Date().toLocaleTimeString();
    } catch (err) {
      console.error('Error fetching health data', err);
      error = err.message || 'Failed to fetch health data';
    } finally {
      loading = false;
    }
  }

  // Toggle auto-refresh
  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      refreshInterval = setInterval(fetchHealthData, 10000); // Refresh every 10 seconds
    } else if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  }

  // Format bytes to human-readable form
  function formatBytes(bytes, decimals = 2) {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  // Calculate status color
  function getStatusColor(status) {
    if (!status) return 'bg-gray-300';
    status = status.toLowerCase();
    if (status === 'ok' || status === 'ready') return 'bg-green-500';
    if (status === 'warning' || status === 'schema_incomplete') return 'bg-yellow-500';
    if (status === 'error' || status === 'connection_failed') return 'bg-red-500';
    return 'bg-gray-300';
  }

  // Initialize on mount
  onMount(() => {
    fetchHealthData();
  });

  // Clean up on destroy
  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="health-dashboard">
  <div class="dashboard-header">
    <h1 class="dashboard-title">System Health Dashboard</h1>

    <div class="dashboard-controls">
      <button class="refresh-button" on:click={fetchHealthData} disabled={loading}>
        <Icon name="refresh" size="18px" class={loading ? 'animate-spin' : ''} />
        Refresh
      </button>

      <label class="auto-refresh-toggle">
        <input type="checkbox" bind:checked={autoRefresh} on:change={toggleAutoRefresh}>
        <span>Auto-refresh</span>
      </label>

      {#if lastUpdated}
        <span class="last-updated">Last updated: {lastUpdated}</span>
      {/if}
    </div>
  </div>

  {#if error}
    <div class="error-message">
      <Icon name="error" size="20px" class="text-red-500" />
      {error}
    </div>
  {/if}

  <div class="dashboard-grid">
    <!-- System Health Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>System Information</h2>
        <div class="status-indicator {getStatusColor('ok')}"></div>
      </div>

      {#if systemHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">OS:</span>
            <span class="info-value">{systemHealth.os}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Python:</span>
            <span class="info-value">{systemHealth.python_version.split(' ')[0]}</span>
          </div>

          <div class="info-item">
            <span class="info-label">CPU Cores:</span>
            <span class="info-value">{systemHealth.cpu_count}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Memory:</span>
            <div class="progress-container">
              <div class="progress-bar" style="width: {systemHealth.memory_percent}%"></div>
              <span class="progress-text">
                {formatBytes(systemHealth.memory_available)} free of {formatBytes(systemHealth.memory_total)}
                ({systemHealth.memory_percent}% used)
              </span>
            </div>
          </div>

          <div class="info-item">
            <span class="info-label">Disk:</span>
            <div class="progress-container">
              <div class="progress-bar" style="width: {systemHealth.disk_usage['/'] || 0}%"></div>
              <span class="progress-text">{systemHealth.disk_usage['/'] || 0}% used</span>
            </div>
          </div>
        </div>
      {:else}
        <div class="loading-placeholder">Loading system information...</div>
      {/if}
    </div>

    <!-- Database Health Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>Database Health</h2>
        {#if databaseHealth}
          <div class="status-indicator {getStatusColor(databaseHealth.status)}"></div>
        {:else}
          <div class="status-indicator bg-gray-300"></div>
        {/if}
      </div>

      {#if databaseHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{databaseHealth.status}">{databaseHealth.status}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Type:</span>
            <span class="info-value">{databaseHealth.type || 'Unknown'}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Tables:</span>
            <div class="table-counts">
              {#each Object.entries(databaseHealth.table_counts) as [table, count]}
                <div class="table-count-item">
                  <span class="table-name">{table}:</span>
                  <span class="table-count">{count}</span>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {:else}
        <div class="loading-placeholder">Loading database information...</div>
      {/if}
    </div>

    <!-- Vector Store Health Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>Vector Store Health</h2>
        {#if vectorStoreHealth}
          <div class="status-indicator {getStatusColor(vectorStoreHealth.status)}"></div>
        {:else}
          <div class="status-indicator bg-gray-300"></div>
        {/if}
      </div>

      {#if vectorStoreHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{vectorStoreHealth.status}">{vectorStoreHealth.status}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Connection:</span>
            <span class="info-value status-{vectorStoreHealth.connection ? 'ok' : 'error'}">
              {vectorStoreHealth.connection ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <div class="info-item">
            <span class="info-label">Schema:</span>
            <span class="info-value status-{vectorStoreHealth.database_ready ? 'ok' : 'warning'}">
              {vectorStoreHealth.database_ready ? 'Ready' : 'Incomplete'}
            </span>
          </div>

          <div class="info-item">
            <span class="info-label">Indexes:</span>
            <span class="info-value">{vectorStoreHealth.indexes_count} total, {vectorStoreHealth.vector_indexes_count} vector</span>
          </div>

          <div class="info-item">
            <span class="info-label">Node Counts:</span>
            <div class="node-counts">
              {#each Object.entries(vectorStoreHealth.node_counts) as [label, count]}
                <div class="node-count-item">
                  <span class="node-label">{label}:</span>
                  <span class="node-count">{count}</span>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {:else}
        <div class="loading-placeholder">Loading vector store information...</div>
      {/if}
    </div>

    <!-- Memory Health Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>Memory Storage Health</h2>
        {#if memoryHealth}
          <div class="status-indicator {getStatusColor(memoryHealth.status)}"></div>
        {:else}
          <div class="status-indicator bg-gray-300"></div>
        {/if}
      </div>

      {#if memoryHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{memoryHealth.status}">{memoryHealth.status}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Conversations:</span>
            <span class="info-value">{memoryHealth.conversation_count} ({memoryHealth.backup_count} backups)</span>
          </div>

          <div class="info-item">
            <span class="info-label">Storage Path:</span>
            <span class="info-value truncate">{memoryHealth.storage_path}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Disk Space:</span>
            <span class="info-value">{memoryHealth.disk_free_mb} MB free</span>
          </div>

          <div class="info-item">
            <span class="info-label">Sample Check:</span>
            <span class="info-value status-{memoryHealth.sample_conversation_status === 'ok' ? 'ok' : 'warning'}">
              {memoryHealth.sample_conversation_status}
            </span>
          </div>
        </div>
      {:else}
        <div class="loading-placeholder">Loading memory information...</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .health-dashboard {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  .dashboard-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a202c;
    margin: 0;
  }

  .dashboard-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .refresh-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background-color: #4a63ee;
    color: white;
    border: none;
    border-radius: 0.375rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .refresh-button:hover {
    background-color: #3a53de;
  }

  .refresh-button:disabled {
    background-color: #a0aec0;
    cursor: not-allowed;
  }

  .auto-refresh-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    font-size: 0.875rem;
    color: #4a5568;
  }

  .last-updated {
    font-size: 0.75rem;
    color: #718096;
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background-color: #fee2e2;
    color: #b91c1c;
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 1.5rem;
    font-size: 0.875rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
    gap: 1.5rem;
  }

  .health-card {
    background-color: white;
    border-radius: 0.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    overflow: hidden;
  }

  .card-header {
    padding: 1rem 1.5rem;
    background-color: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .card-header h2 {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
    color: #2d3748;
  }

  .status-indicator {
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 50%;
  }

  .card-content {
    padding: 1.5rem;
  }

  .info-item {
    margin-bottom: 1rem;
  }

  .info-item:last-child {
    margin-bottom: 0;
  }

  .info-label {
    display: block;
    font-size: 0.875rem;
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 0.25rem;
  }

  .info-value {
    display: block;
    font-size: 0.875rem;
    color: #2d3748;
  }

  .status-ok {
    color: #059669;
  }

  .status-warning {
    color: #d97706;
  }

  .status-error {
    color: #dc2626;
  }

  .progress-container {
    background-color: #e2e8f0;
    height: 0.5rem;
    border-radius: 0.25rem;
    overflow: hidden;
    margin-top: 0.25rem;
    position: relative;
  }

  .progress-bar {
    background-color: #4a63ee;
    height: 100%;
    border-radius: 0.25rem;
  }

  .progress-text {
    display: block;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
  }

  .table-counts, .node-counts {
    margin-top: 0.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 0.5rem;
  }

  .table-count-item, .node-count-item {
    font-size: 0.75rem;
    color: #4b5563;
  }

  .table-name, .node-label {
    font-weight: 500;
  }

  .loading-placeholder {
    padding: 2rem;
    text-align: center;
    color: #a0aec0;
    font-size: 0.875rem;
  }

  .animate-spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .truncate {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
  }

  @media (max-width: 768px) {
    .health-dashboard {
      padding: 1rem;
    }

    .dashboard-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }

    .dashboard-grid {
      grid-template-columns: 1fr;
    }
  }
</style>

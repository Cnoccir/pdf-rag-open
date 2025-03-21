<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { api } from '$api';
  import { health } from '../api/health';  // Direct import from health.ts
  import Icon from '$c/Icon.svelte';

  // State variables
  let systemHealth = null;
  let databaseHealth = null;
  let vectorStoreHealth = null;
  let memoryHealth = null;
  let metricsData = null;
  let loading = {
    system: true,
    database: true,
    vector: true,
    memory: true,
    metrics: true,
    pdf: false,
    query: false
  };
  let error = {
    system: null,
    database: null,
    vector: null,
    memory: null,
    metrics: null,
    pdf: null,
    query: null
  };
  let refreshInterval = null;
  let autoRefresh = false;
  let lastUpdated = '';

  // PDF Health Check
  let selectedPdfId = '';
  let pdfHealth = null;

  // Query Test
  let queryText = 'What is this document about?';
  let queryResults = null;
  let queryPdfId = '';

  // Fetch health data
  async function fetchHealthData() {
    // Reset last errors
    error.system = null;
    error.database = null;
    error.vector = null;
    error.memory = null;
    error.metrics = null;

    // System health
    loading.system = true;
    try {
      const res = await health.getSystemHealth();
      systemHealth = res.data;
    } catch (err) {
      console.error('Error fetching system health', err);
      error.system = err.message || 'Failed to fetch system health';
    } finally {
      loading.system = false;
    }

    // Database health - note the updated endpoint to plural "databases"
    loading.database = true;
    try {
      const res = await health.getDatabaseHealth();
      databaseHealth = res.data;
    } catch (err) {
      console.error('Error fetching database health', err);
      error.database = err.message || 'Failed to fetch database health';
    } finally {
      loading.database = false;
    }

    // Vector store health - note the updated endpoint
    loading.vector = true;
    try {
      const res = await api.get('/health/vector_stores');
      vectorStoreHealth = res.data;
    } catch (err) {
      console.error('Error fetching vector store health', err);
      error.vector = err.message || 'Failed to fetch vector store health';
    } finally {
      loading.vector = false;
    }

    // Memory health
    loading.memory = true;
    try {
      const res = await api.get('/health/memory');
      memoryHealth = res.data;
    } catch (err) {
      console.error('Error fetching memory health', err);
      error.memory = err.message || 'Failed to fetch memory health';
    } finally {
      loading.memory = false;
    }

    // Metrics data - new endpoint
    loading.metrics = true;
    try {
      const res = await api.get('/health/metrics');
      metricsData = res.data;
    } catch (err) {
      console.error('Error fetching metrics', err);
      error.metrics = err.message || 'Failed to fetch metrics';
    } finally {
      loading.metrics = false;
    }

    lastUpdated = new Date().toLocaleTimeString();
  }

  // Fetch PDF health
  async function fetchPdfHealth() {
    if (!selectedPdfId) return;

    loading.pdf = true;
    error.pdf = null;
    pdfHealth = null;

    try {
      const res = await health.getPdfHealth(selectedPdfId);
      pdfHealth = res.data;
    } catch (err) {
      console.error('Error fetching PDF health', err);
      error.pdf = err.message || 'Failed to fetch PDF health';
    } finally {
      loading.pdf = false;
    }
  }

  // Run test query
  async function runTestQuery() {
    if (!queryPdfId) {
      // Use the selected PDF ID if query PDF ID is not specified
      queryPdfId = selectedPdfId;
      if (!queryPdfId) {
        error.query = "Please enter a PDF ID for the query";
        return;
      }
    }

    loading.query = true;
    error.query = null;
    queryResults = null;

    try {
      const res = await health.runTestQuery(queryText, queryPdfId);
      queryResults = res.data;
    } catch (err) {
      console.error('Error running test query', err);
      error.query = err.message || 'Failed to run test query';
    } finally {
      loading.query = false;
    }
  }

  // Reinitialize vector stores
  async function reinitializeStores() {
    if (!confirm('Are you sure you want to reinitialize all vector stores? This may take a moment.')) {
      return;
    }

    loading.database = true;
    error.database = null;

    try {
      const res = await api.get('/health/databases?force_init=true');
      databaseHealth = res.data;
      alert('Vector stores reinitialized successfully');
    } catch (err) {
      console.error('Error reinitializing vector stores', err);
      error.database = err.message || 'Failed to reinitialize vector stores';
    } finally {
      loading.database = false;
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
    if (status === 'warning' || status === 'degraded' || status === 'partial' || status === 'schema_incomplete') return 'bg-yellow-500';
    if (status === 'error' || status === 'connection_failed' || status === 'not_found') return 'bg-red-500';
    return 'bg-gray-300';
  }

  // Format timestamp
  function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch (e) {
      return timestamp;
    }
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
      <button class="refresh-button" on:click={fetchHealthData} disabled={loading.system || loading.database || loading.vector || loading.memory}>
        <Icon name="refresh" size="18px" class={loading.system || loading.database || loading.vector || loading.memory ? 'animate-spin' : ''} />
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

  <!-- Health Cards Grid -->
  <div class="dashboard-grid">
    <!-- System Health Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>System Information</h2>
        <div class="status-indicator {getStatusColor(systemHealth?.status || 'loading')}"></div>
      </div>

      {#if loading.system}
        <div class="loading-placeholder">Loading system information...</div>
      {:else if error.system}
        <div class="error-message">
          <Icon name="error" size="20px" class="text-red-500" />
          {error.system}
        </div>
      {:else if systemHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{systemHealth.status || 'unknown'}">{systemHealth.status || 'Unknown'}</span>
          </div>

          {#if systemHealth.components}
            <div class="info-item">
              <span class="info-label">Components:</span>
              <div class="components-list">
                {#each Object.entries(systemHealth.components) as [name, component]}
                  <div class="component-item">
                    <span class="component-name">{name}:</span>
                    <span class="component-status status-{component.status || 'unknown'}">{component.status || 'Unknown'}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          {#if systemHealth.os}
            <div class="info-item">
              <span class="info-label">OS:</span>
              <span class="info-value">{systemHealth.os}</span>
            </div>
          {/if}

          {#if systemHealth.python_version}
            <div class="info-item">
              <span class="info-label">Python:</span>
              <span class="info-value">{systemHealth.python_version.split(' ')[0]}</span>
            </div>
          {/if}

          {#if systemHealth.cpu_count}
            <div class="info-item">
              <span class="info-label">CPU Cores:</span>
              <span class="info-value">{systemHealth.cpu_count}</span>
            </div>
          {/if}

          {#if systemHealth.memory_available}
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
          {/if}

          {#if systemHealth.disk_usage}
            <div class="info-item">
              <span class="info-label">Disk:</span>
              <div class="progress-container">
                <div class="progress-bar" style="width: {systemHealth.disk_usage['/'] || 0}%"></div>
                <span class="progress-text">{systemHealth.disk_usage['/'] || 0}% used</span>
              </div>
            </div>
          {/if}
        </div>
      {:else}
        <div class="loading-placeholder">No system information available</div>
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

      {#if loading.database}
        <div class="loading-placeholder">Loading database information...</div>
      {:else if error.database}
        <div class="error-message">
          <Icon name="error" size="20px" class="text-red-500" />
          {error.database}
        </div>
      {:else if databaseHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{databaseHealth.status}">{databaseHealth.status}</span>
          </div>

          {#if databaseHealth.databases}
            <div class="info-item">
              <span class="info-label">Databases:</span>
              <div class="database-list">
                {#each Object.entries(databaseHealth.databases) as [name, db]}
                  <div class="database-item">
                    <span class="database-name">{name}:</span>
                    <span class="database-status status-{db.status || 'unknown'}">{db.status || 'Unknown'}</span>
                    <span class="database-initialized">{db.initialized ? '✓' : '✗'}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          {#if databaseHealth.all_initialized !== undefined}
            <div class="info-item">
              <span class="info-label">All Initialized:</span>
              <span class="info-value status-{databaseHealth.all_initialized ? 'ok' : 'warning'}">
                {databaseHealth.all_initialized ? 'Yes' : 'No'}
              </span>
            </div>
          {/if}

          {#if databaseHealth.mongo && databaseHealth.mongo.stats}
            <div class="info-item">
              <span class="info-label">MongoDB Collections:</span>
              <div class="table-counts">
                {#each Object.entries(databaseHealth.mongo.stats.collection_counts || {}) as [collection, count]}
                  <div class="table-count-item">
                    <span class="table-name">{collection}:</span>
                    <span class="table-count">{count}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}

          <div class="info-item">
            <button class="action-button" on:click={reinitializeStores} disabled={loading.database}>
              Reinitialize Vector Stores
            </button>
          </div>
        </div>
      {:else}
        <div class="loading-placeholder">No database information available</div>
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

      {#if loading.vector}
        <div class="loading-placeholder">Loading vector store information...</div>
      {:else if error.vector}
        <div class="error-message">
          <Icon name="error" size="20px" class="text-red-500" />
          {error.vector}
        </div>
      {:else if vectorStoreHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{vectorStoreHealth.status}">{vectorStoreHealth.status}</span>
          </div>

          {#if vectorStoreHealth.mongo}
            <div class="info-item">
              <span class="info-label">MongoDB:</span>
              <span class="info-value status-{vectorStoreHealth.mongo.status || 'unknown'}">
                {vectorStoreHealth.mongo.status || 'Unknown'}
              </span>
            </div>
          {/if}

          {#if vectorStoreHealth.qdrant}
            <div class="info-item">
              <span class="info-label">Qdrant:</span>
              <span class="info-value status-{vectorStoreHealth.qdrant.status || 'unknown'}">
                {vectorStoreHealth.qdrant.status || 'Unknown'}
              </span>
              {#if vectorStoreHealth.qdrant.vector_count !== undefined}
                <span class="vector-count">({vectorStoreHealth.qdrant.vector_count} vectors)</span>
              {/if}
            </div>
          {/if}

          {#if vectorStoreHealth.unified}
            <div class="info-item">
              <span class="info-label">Unified Store:</span>
              <span class="info-value status-{vectorStoreHealth.unified.status || 'unknown'}">
                {vectorStoreHealth.unified.status || 'Unknown'}
              </span>
              {#if vectorStoreHealth.unified.embedding_model}
                <span class="model-info">Model: {vectorStoreHealth.unified.embedding_model}</span>
              {/if}
            </div>
          {/if}

          {#if vectorStoreHealth.metrics}
            <div class="info-item">
              <span class="info-label">Metrics:</span>
              <div class="metrics-list">
                <div class="metric-item">
                  <span class="metric-name">Queries:</span>
                  <span class="metric-value">{vectorStoreHealth.metrics.queries || 0}</span>
                </div>
                <div class="metric-item">
                  <span class="metric-name">Errors:</span>
                  <span class="metric-value">{vectorStoreHealth.metrics.errors || 0}</span>
                </div>
                {#if vectorStoreHealth.metrics.avg_query_time}
                  <div class="metric-item">
                    <span class="metric-name">Avg Query Time:</span>
                    <span class="metric-value">{(vectorStoreHealth.metrics.avg_query_time).toFixed(2)}s</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}
        </div>
      {:else}
        <div class="loading-placeholder">No vector store information available</div>
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

      {#if loading.memory}
        <div class="loading-placeholder">Loading memory information...</div>
      {:else if error.memory}
        <div class="error-message">
          <Icon name="error" size="20px" class="text-red-500" />
          {error.memory}
        </div>
      {:else if memoryHealth}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Status:</span>
            <span class="info-value status-{memoryHealth.status}">{memoryHealth.status}</span>
          </div>

          <div class="info-item">
            <span class="info-label">Conversations:</span>
            <span class="info-value">{memoryHealth.conversation_count || 0} ({memoryHealth.backup_count || 0} backups)</span>
          </div>

          {#if memoryHealth.storage_path}
            <div class="info-item">
              <span class="info-label">Storage Path:</span>
              <span class="info-value truncate">{memoryHealth.storage_path}</span>
            </div>
          {/if}

          {#if memoryHealth.disk_free_mb}
            <div class="info-item">
              <span class="info-label">Disk Space:</span>
              <span class="info-value">{memoryHealth.disk_free_mb} MB free</span>
            </div>
          {/if}

          {#if memoryHealth.sample_conversation_status}
            <div class="info-item">
              <span class="info-label">Sample Check:</span>
              <span class="info-value status-{memoryHealth.sample_conversation_status === 'ok' ? 'ok' : 'warning'}">
                {memoryHealth.sample_conversation_status}
              </span>
            </div>
          {/if}
        </div>
      {:else}
        <div class="loading-placeholder">No memory information available</div>
      {/if}
    </div>

    <!-- Metrics Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>System Metrics</h2>
        {#if metricsData}
          <div class="status-indicator bg-blue-500"></div>
        {:else}
          <div class="status-indicator bg-gray-300"></div>
        {/if}
      </div>

      {#if loading.metrics}
        <div class="loading-placeholder">Loading metrics information...</div>
      {:else if error.metrics}
        <div class="error-message">
          <Icon name="error" size="20px" class="text-red-500" />
          {error.metrics}
        </div>
      {:else if metricsData}
        <div class="card-content">
          <div class="info-item">
            <span class="info-label">Last Updated:</span>
            <span class="info-value">{formatTimestamp(metricsData.timestamp)}</span>
          </div>

          <!-- PDF Metrics -->
          {#if metricsData.pdfs}
            <div class="info-item">
              <span class="info-label">PDFs:</span>
              <div class="metrics-grid">
                <div class="metric-box">
                  <span class="metric-value">{metricsData.pdfs.total || 0}</span>
                  <span class="metric-label">Total</span>
                </div>
                <div class="metric-box">
                  <span class="metric-value">{metricsData.pdfs.active || 0}</span>
                  <span class="metric-label">Active</span>
                </div>
                <div class="metric-box">
                  <span class="metric-value">{metricsData.pdfs.processed || 0}</span>
                  <span class="metric-label">Processed</span>
                </div>
                {#if metricsData.pdfs.with_errors !== undefined}
                  <div class="metric-box">
                    <span class="metric-value">{metricsData.pdfs.with_errors}</span>
                    <span class="metric-label">Errors</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}

          <!-- Conversation Metrics -->
          {#if metricsData.conversations}
            <div class="info-item">
              <span class="info-label">Conversations:</span>
              <div class="metrics-grid">
                <div class="metric-box">
                  <span class="metric-value">{metricsData.conversations.total || 0}</span>
                  <span class="metric-label">Total</span>
                </div>
                <div class="metric-box">
                  <span class="metric-value">{metricsData.conversations.active || 0}</span>
                  <span class="metric-label">Active</span>
                </div>
                {#if metricsData.conversations.messages !== undefined}
                  <div class="metric-box">
                    <span class="metric-value">{metricsData.conversations.messages}</span>
                    <span class="metric-label">Messages</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}

          <!-- Qdrant Metrics -->
          {#if metricsData.qdrant}
            <div class="info-item">
              <span class="info-label">Qdrant:</span>
              <div class="metrics-grid">
                <div class="metric-box">
                  <span class="metric-value">{metricsData.qdrant.vector_count || 0}</span>
                  <span class="metric-label">Vectors</span>
                </div>
                {#if metricsData.qdrant.metrics}
                  <div class="metric-box">
                    <span class="metric-value">{metricsData.qdrant.metrics.queries || 0}</span>
                    <span class="metric-label">Queries</span>
                  </div>
                  <div class="metric-box">
                    <span class="metric-value">{(metricsData.qdrant.metrics.avg_query_time || 0).toFixed(2)}s</span>
                    <span class="metric-label">Avg Time</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}

          <!-- RAG Monitor Stats -->
          {#if metricsData.rag_monitor && metricsData.rag_monitor.operation_types}
            <div class="info-item">
              <span class="info-label">Recent Operations:</span>
              <div class="bar-chart">
                {#each Object.entries(metricsData.rag_monitor.operation_types) as [type, count]}
                  <div class="bar-container">
                    <span class="bar-label">{type}</span>
                    <div class="bar" style="width: {Math.min(count * 10, 100)}%;">{count}</div>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {:else}
        <div class="loading-placeholder">No metrics information available</div>
      {/if}
    </div>

    <!-- PDF Health Check Card -->
    <div class="health-card">
      <div class="card-header">
        <h2>PDF Health Check</h2>
        {#if pdfHealth}
          <div class="status-indicator {getStatusColor(pdfHealth.status)}"></div>
        {:else}
          <div class="status-indicator bg-gray-300"></div>
        {/if}
      </div>

      <div class="card-content">
        <div class="pdf-search">
          <input
            type="text"
            placeholder="Enter PDF ID"
            bind:value={selectedPdfId}
            class="pdf-input"
          />
          <button
            class="action-button"
            on:click={fetchPdfHealth}
            disabled={!selectedPdfId || loading.pdf}
          >
            {loading.pdf ? 'Checking...' : 'Check'}
          </button>
        </div>

        {#if loading.pdf}
          <div class="loading-placeholder">Checking PDF health...</div>
        {:else if error.pdf}
          <div class="error-message">
            <Icon name="error" size="20px" class="text-red-500" />
            {error.pdf}
          </div>
        {:else if pdfHealth}
          <div class="pdf-details">
            <div class="info-item">
              <span class="info-label">Status:</span>
              <span class="info-value status-{pdfHealth.status}">{pdfHealth.status}</span>
            </div>

            <div class="info-item">
              <span class="info-label">PDF ID:</span>
              <span class="info-value">{pdfHealth.pdf_id}</span>
            </div>

            <div class="info-item">
              <span class="info-label">Title:</span>
              <span class="info-value">{pdfHealth.document_title || 'Unknown'}</span>
            </div>

            <div class="info-item">
              <span class="info-label">Storage:</span>
              <div class="storage-status">
                <div class="storage-item">
                  <span class="storage-name">MongoDB:</span>
                  <span class="storage-indicator {pdfHealth.exists_in_mongo ? 'bg-green-500' : 'bg-red-500'}"></span>
                </div>
                <div class="storage-item">
                  <span class="storage-name">Qdrant:</span>
                  <span class="storage-indicator {pdfHealth.exists_in_qdrant ? 'bg-green-500' : 'bg-red-500'}"></span>
                </div>
                <div class="storage-item">
                  <span class="storage-name">S3:</span>
                  <span class="storage-indicator {pdfHealth.exists_in_s3 ? 'bg-green-500' : 'bg-red-500'}"></span>
                </div>
              </div>
            </div>

            <div class="info-item">
              <span class="info-label">Elements:</span>
              <span class="info-value">{pdfHealth.element_count || 0} elements, {pdfHealth.embedding_count || 0} embeddings</span>
            </div>

            {#if pdfHealth.content_type_breakdown}
              <div class="info-item">
                <span class="info-label">Content Types:</span>
                <div class="bar-chart">
                  {#each Object.entries(pdfHealth.content_type_breakdown) as [type, count]}
                    <div class="bar-container">
                      <span class="bar-label">{type}</span>
                      <div class="bar" style="width: {Math.min(count * 5, 100)}%;">{count}</div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            {#if pdfHealth.exists_in_qdrant}
              <div class="info-item">
                <span class="info-label">Test Query:</span>
                <div class="query-form">
                  <input
                    type="text"
                    placeholder="Query text"
                    bind:value={queryText}
                    class="query-input"
                  />
                  <button
                    class="action-button"
                    on:click={() => {
                      queryPdfId = pdfHealth.pdf_id;
                      runTestQuery();
                    }}
                    disabled={loading.query}
                  >
                    {loading.query ? 'Running...' : 'Run Query'}
                  </button>
                </div>
              </div>
            {/if}

            {#if queryResults && queryPdfId === pdfHealth.pdf_id}
              <div class="info-item">
                <span class="info-label">Query Results:</span>
                <div class="query-results">
                  <div class="query-stats">
                    <span>Found {queryResults.results_count} results in {queryResults.time_taken.toFixed(2)}s</span>
                  </div>
                  {#if queryResults.results && queryResults.results.length > 0}
                    <div class="results-list">
                      {#each queryResults.results as result, i}
                        <div class="result-item">
                          <div class="result-header">
                            <span class="result-index">#{i+1}</span>
                            <span class="result-score">Score: {result.score.toFixed(4)}</span>
                          </div>
                          <div class="result-content">{result.content}</div>
                          <div class="result-meta">
                            <span>Type: {result.content_type}</span>
                            <span>Page: {result.page}</span>
                          </div>
                        </div>
                      {/each}
                    </div>
                  {:else}
                    <div class="no-results">No results found</div>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        {:else if selectedPdfId}
          <div class="pdf-placeholder">Click "Check" to verify this PDF ID</div>
        {:else}
          <div class="pdf-placeholder">Enter a PDF ID to check its health and run test queries</div>
        {/if}
      </div>
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
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
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

  .status-warning, .status-degraded, .status-partial {
    color: #d97706;
  }

  .status-error, .status-not_found {
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

  .components-list, .database-list {
    margin-top: 0.5rem;
  }

  .component-item, .database-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
    font-size: 0.875rem;
  }

  .component-name, .database-name {
    font-weight: 500;
    color: #4b5563;
  }

  .database-initialized {
    color: #059669;
    font-weight: 600;
  }

  .metrics-list {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.5rem;
  }

  .metric-item {
    font-size: 0.875rem;
    color: #4b5563;
  }

  .metric-name {
    font-weight: 500;
    margin-right: 0.25rem;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
  }

  .metric-box {
    background-color: #f1f5f9;
    border-radius: 0.375rem;
    padding: 0.75rem 0.5rem;
    text-align: center;
  }

  .metric-value {
    display: block;
    font-size: 1.125rem;
    font-weight: 600;
    color: #334155;
  }

  .metric-label {
    display: block;
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
  }

  .loading-placeholder {
    padding: 2rem;
    text-align: center;
    color: #a0aec0;
    font-size: 0.875rem;
  }

  .pdf-search {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .pdf-input, .query-input {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  .action-button {
    padding: 0.5rem 1rem;
    background-color: #4a63ee;
    color: white;
    border: none;
    border-radius: 0.375rem;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
    white-space: nowrap;
  }

  .action-button:hover {
    background-color: #3a53de;
  }

  .action-button:disabled {
    background-color: #a0aec0;
    cursor: not-allowed;
  }

  .pdf-placeholder {
    text-align: center;
    padding: 2rem 0;
    color: #a0aec0;
    font-size: 0.875rem;
  }

  .storage-status {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.5rem;
  }

  .storage-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .storage-name {
    font-size: 0.875rem;
    color: #4b5563;
  }

  .storage-indicator {
    width: 0.625rem;
    height: 0.625rem;
    border-radius: 50%;
  }

  .bar-chart {
    margin-top: 0.5rem;
  }

  .bar-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .bar-label {
    min-width: 80px;
    font-size: 0.75rem;
    color: #4b5563;
  }

  .bar {
    height: 1.25rem;
    background-color: #4a63ee;
    color: white;
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    padding: 0 0.5rem;
    border-radius: 0.25rem;
    transition: width 0.3s;
    min-width: 1.5rem;
  }

  .query-form {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .query-stats {
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    color: #4b5563;
  }

  .results-list {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #e2e8f0;
    border-radius: 0.375rem;
  }

  .result-item {
    padding: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
  }

  .result-item:last-child {
    border-bottom: none;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.25rem;
  }

  .result-index {
    font-weight: 600;
    font-size: 0.75rem;
    color: #4a5568;
  }

  .result-score {
    font-size: 0.75rem;
    color: #4a63ee;
  }

  .result-content {
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
    color: #1a202c;
    line-height: 1.4;
  }

  .result-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
    color: #64748b;
  }

  .no-results {
    padding: 1rem;
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

// src/api/index.ts

import { api, getErrorMessage, getError, triggerEmbeddingProcess,
         deletePdf, activateResearchMode, deactivateResearchMode } from './axios';
import { health } from './health';

// Re-export everything
export {
  api,
  getErrorMessage,
  getError,
  triggerEmbeddingProcess,
  deletePdf,
  activateResearchMode,
  deactivateResearchMode,
  health
};

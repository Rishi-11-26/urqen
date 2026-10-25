/**
 * api.js
 * Centralized API client for URQEN backend communication.
 * Includes Supabase JWT bearer token on every request when user is signed in.
 */

const BASE_URL = '/api';

// Token provider — App.jsx sets this when session changes
let _getToken = () => null;
export const setTokenProvider = (fn) => { _getToken = fn; };

function authHeaders(extra = {}) {
  const token = _getToken();
  return token
    ? { Authorization: `Bearer ${token}`, ...extra }
    : { ...extra };
}

async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = res.statusText;
    const responseBody = await res.text();
    if (responseBody) {
      try {
        const errJson = JSON.parse(responseBody);
        errorDetail = errJson.detail || JSON.stringify(errJson);
      } catch {
        errorDetail = responseBody;
      }
    }
    throw new Error(errorDetail || `Request failed with status ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Health
  checkHealth: async () => {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  },

  // Storage Quota
  getStorageQuota: async () => {
    const res = await fetch(`${BASE_URL}/datasets/storage/quota`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  // Datasets
  listDatasets: async () => {
    const res = await fetch(`${BASE_URL}/datasets`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  uploadDataset: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${BASE_URL}/datasets/upload`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });
    return handleResponse(res);
  },

  getDatasetProfile: async (datasetId, sampleRows = 25) => {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}/profile?sample_rows=${sampleRows}`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getDatasetAudit: async (datasetId, targetColumn = null) => {
    let url = `${BASE_URL}/datasets/${datasetId}/audit`;
    if (targetColumn) {
      url += `?target_column=${encodeURIComponent(targetColumn)}`;
    }
    const res = await fetch(url, { headers: authHeaders() });
    return handleResponse(res);
  },

  deleteDataset: async (datasetId) => {
    const res = await fetch(`${BASE_URL}/datasets/${datasetId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  // Experiments
  listExperiments: async () => {
    const res = await fetch(`${BASE_URL}/experiments`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  createExperiment: async (payload) => {
    const res = await fetch(`${BASE_URL}/experiments/benchmark`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  getExperiment: async (experimentId) => {
    const res = await fetch(`${BASE_URL}/experiments/${experimentId}`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  cancelExperiment: async (experimentId) => {
    const res = await fetch(`${BASE_URL}/experiments/${experimentId}/cancel`, {
      method: 'POST',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  deleteExperiment: async (experimentId) => {
    const res = await fetch(`${BASE_URL}/experiments/${experimentId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  // Artifacts & Inference
  downloadReportUrl: (experimentId, format = 'html') => {
    const t = typeof _getToken === 'function' ? _getToken() : null;
    const tokenParam = t ? `&token=${encodeURIComponent(t)}` : '';
    return `${BASE_URL}/artifacts/${experimentId}/report?format=${format}${tokenParam}`;
  },

  downloadManifestUrl: (experimentId) => {
    const t = typeof _getToken === 'function' ? _getToken() : null;
    const tokenParam = t ? `?token=${encodeURIComponent(t)}` : '';
    return `${BASE_URL}/artifacts/${experimentId}/manifest${tokenParam}`;
  },

  downloadEvaluationUrl: (experimentId) => {
    const t = typeof _getToken === 'function' ? _getToken() : null;
    const tokenParam = t ? `?token=${encodeURIComponent(t)}` : '';
    return `${BASE_URL}/artifacts/${experimentId}/evaluation${tokenParam}`;
  },

  downloadBundleUrl: (experimentId) => {
    const t = typeof _getToken === 'function' ? _getToken() : null;
    const tokenParam = t ? `?token=${encodeURIComponent(t)}` : '';
    return `${BASE_URL}/artifacts/${experimentId}/model-bundle${tokenParam}`;
  },

  downloadPreprocessedDataUrl: (experimentId) => {
    const t = typeof _getToken === 'function' ? _getToken() : null;
    const tokenParam = t ? `?token=${encodeURIComponent(t)}` : '';
    return `${BASE_URL}/artifacts/${experimentId}/preprocessed-dataset${tokenParam}`;
  },

  runInference: async (experimentId, rows) => {
    const res = await fetch(`${BASE_URL}/artifacts/${experimentId}/predict`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ rows }),
    });
    return handleResponse(res);
  },

  // Admin Console
  checkAdminStatus: async () => {
    const res = await fetch(`${BASE_URL}/admin/check`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  sendPresenceHeartbeat: async () => {
    const res = await fetch(`${BASE_URL}/presence/heartbeat`, {
      method: 'POST',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAccountNotice: async () => {
    const res = await fetch(`${BASE_URL}/account/notice`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAdminOverview: async () => {
    const res = await fetch(`${BASE_URL}/admin/overview`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAdminUsers: async () => {
    const res = await fetch(`${BASE_URL}/admin/users`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAdminLiveUsers: async () => {
    const res = await fetch(`${BASE_URL}/admin/live-users`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAdminExperiments: async () => {
    const res = await fetch(`${BASE_URL}/admin/experiments`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  getAdminDatasets: async () => {
    const res = await fetch(`${BASE_URL}/admin/datasets`, {
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  downloadAdminHistoryPdf: async () => {
    const res = await fetch(`${BASE_URL}/admin/history.pdf`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      return handleResponse(res);
    }
    return res.blob();
  },

  purgeRetentionArtifacts: async (maxAgeDays = 14) => {
    const res = await fetch(`${BASE_URL}/admin/purge-retention`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ max_age_days: maxAgeDays }),
    });
    return handleResponse(res);
  },

  purgeAllUserData: async (confirmText) => {
    const res = await fetch(`${BASE_URL}/admin/purge-all`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ confirm_text: confirmText }),
    });
    return handleResponse(res);
  },

  updateUserQuota: async (userId, quotaMb) => {
    const res = await fetch(`${BASE_URL}/admin/users/${userId}/quota`, {
      method: 'PUT',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ quota_mb: quotaMb }),
    });
    return handleResponse(res);
  },

  adminDeleteExperiment: async (experimentId) => {
    const res = await fetch(`${BASE_URL}/admin/experiments/${experimentId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  adminDeleteDataset: async (datasetId) => {
    const res = await fetch(`${BASE_URL}/admin/datasets/${datasetId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },

  adminDeleteUser: async (userId) => {
    const res = await fetch(`${BASE_URL}/admin/users/${userId}`, {
      method: 'DELETE',
      headers: authHeaders(),
    });
    return handleResponse(res);
  },
};

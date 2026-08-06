/**
 * Stateless message router. All stateful logic (camera, ML, verification)
 * lives in the monitor page; the service worker only:
 *  - opens the monitor window from the toolbar button,
 *  - re-broadcasts messages between the content script and the monitor page,
 *  - tracks the monitor heartbeat and shows it on the action badge.
 */
import type { RuntimeMessage, MonitorStatusReply } from '../shared/messages';

let lastHeartbeatTs = 0;
let lastCameraOk = false;
let monitorWindowId: number | null = null;

const HEARTBEAT_STALE_MS = 5000;

function monitorAlive(): boolean {
  return Date.now() - lastHeartbeatTs < HEARTBEAT_STALE_MS;
}

function updateBadge() {
  const alive = monitorAlive();
  const ok = alive && lastCameraOk;
  chrome.action.setBadgeText({ text: ok ? 'ON' : 'OFF' });
  chrome.action.setBadgeBackgroundColor({ color: ok ? '#16a34a' : '#dc2626' });
}

async function openMonitor() {
  if (monitorWindowId !== null) {
    try {
      await chrome.windows.update(monitorWindowId, { focused: true });
      return;
    } catch {
      monitorWindowId = null; // window was closed
    }
  }
  const win = await chrome.windows.create({
    url: chrome.runtime.getURL('monitor.html'),
    type: 'popup',
    width: 1000,
    height: 760,
  });
  monitorWindowId = win?.id ?? null;
}

chrome.action.onClicked.addListener(() => {
  void openMonitor();
});

chrome.windows.onRemoved.addListener((id) => {
  if (id === monitorWindowId) {
    monitorWindowId = null;
    lastHeartbeatTs = 0;
    updateBadge();
  }
});

chrome.runtime.onMessage.addListener((message: RuntimeMessage, _sender, sendResponse) => {
  switch (message.type) {
    case 'MONITOR_HEARTBEAT': {
      lastHeartbeatTs = message.ts;
      lastCameraOk = message.cameraOk;
      updateBadge();
      return false;
    }
    case 'MONITOR_STATUS_QUERY': {
      const reply: MonitorStatusReply = {
        type: 'MONITOR_STATUS_REPLY',
        monitorAlive: monitorAlive(),
        cameraOk: lastCameraOk,
      };
      sendResponse(reply);
      return false;
    }
    case 'VERIFICATION_RESULT': {
      // runtime.sendMessage from the monitor already reaches every extension
      // page, but content scripts only receive tabs.sendMessage — forward.
      void chrome.tabs.query({}).then((tabs) => {
        for (const tab of tabs) {
          if (tab.id !== undefined) {
            void chrome.tabs.sendMessage(tab.id, message).catch(() => {});
          }
        }
      });
      return false;
    }
    case 'SCAN_EVENT':
    case 'CONFIG_UPDATED':
      // Delivered to the monitor page directly by runtime messaging; nothing to route.
      return false;
    default:
      return false;
  }
});

updateBadge();

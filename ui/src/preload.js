const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld('electronAPI', {
  displayMenu : (pos) => ipcRenderer.send('menu:display', pos),
  closeWindow : () => ipcRenderer.send('window:close'),
  minimizeWindow : () => ipcRenderer.send('window:minimize'),
  maximizeWindow : () => ipcRenderer.send('window:maximize'),
  unmaximizeWindow : () => ipcRenderer.send('window:unmaximize'),
  maxUnmaxWindow: () => ipcRenderer.send('window:max-unmax'),
  isWindowMaximized: () => ipcRenderer.invoke('window:is-maximized'),
  openFile: () => ipcRenderer.invoke('dialog:open-file'),
  listDir: (dir) => ipcRenderer.invoke('os:list-dir', dir),
  readFile: (path) => ipcRenderer.invoke('os:read-file', path)
})
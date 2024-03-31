const { app, ipcMain, dialog, BrowserWindow, Menu } = require('electron');
const path = require('node:path');
const fs = require('fs');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

async function handleIsWindowMaximized( e ) {
  return BrowserWindow.fromWebContents( e.sender ).isMaximized();
}


async function handleFileOpen () {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openDirectory']
  });
  dialog.show;
  var ret = [filePaths[0]];
  const files = await fs.promises.readdir( filePaths[0] );
  ret = ret.concat(files);
  if (!canceled) {
    return ret;
  }
}

const createWindow = () => {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 600,
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false  
    }
  });

  // and load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }

  // set the application menu
  const menu = Menu.buildFromTemplate([
    {
      label: "文件",
      submenu: [],
    }
  ]);
  Menu.setApplicationMenu(menu);

  // IPC
  ipcMain.on('menu:display', (e, pos) => {
    menu.popup({
      window: BrowserWindow.fromWebContents(e.sender),
      x: pos.x,
      y: pos.y
    })
  });
  ipcMain.on('window:close', (e) => {
    BrowserWindow.fromWebContents(e.sender).close();
  });
  ipcMain.on('window:minimize', (e) => {
    const win = BrowserWindow.fromWebContents(e.sender);
    if (win.minimizable) win.minimize();
  });
  ipcMain.on('window:maximize', (e) => {
    const win = BrowserWindow.fromWebContents(e.sender);
    if (win.maximizable) win.maximize();
  });
  ipcMain.on('window:unmaximize', (e) => {
    BrowserWindow.fromWebContents(e.sender).unmaximize();
  });
  ipcMain.on('window:max-unmax', (e) => {
    const win = BrowserWindow.fromWebContents(e.sender);
    if (win.isMaximized()) {
      win.unmaximize();
    } else {
      win.maximize();
    }
  });

  // Default maximize
  mainWindow.maximize();

  // Open the DevTools.
  mainWindow.webContents.openDevTools();

};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  ipcMain.handle('window:is-maximized', handleIsWindowMaximized);
  ipcMain.handle('dialog:open-file', handleFileOpen);
  createWindow();
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });

});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

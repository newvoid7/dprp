import { handleImportBtnClick, handleRefreshBtnClick } from './project-panel';

const importBtn = document.getElementById('import-btn');
const refreshBtn = document.getElementById('refresh-btn');
const backwardBtn = document.getElementById('backward-btn');
const playBtn = document.getElementById('play-btn');
const forwardBtn = document.getElementById('forward-btn');

importBtn.addEventListener('click', handleImportBtnClick);
refreshBtn.addEventListener('click', handleRefreshBtnClick);

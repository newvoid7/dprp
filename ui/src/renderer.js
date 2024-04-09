import './index.css';

const menuButton = document.getElementById("menu-btn");
const minimizeButton = document.getElementById("minimize-btn");
const maxUnmaxButton = document.getElementById("max-unmax-btn");
const closeButton = document.getElementById("close-btn");


menuButton.addEventListener('click', e => {
	window.electronAPI.displayMenu(e.x, e.y);
});

minimizeButton.addEventListener('click', () => {
	window.electronAPI.minimizeWindow();
});

maxUnmaxButton.addEventListener('click', async () => {
	window.electronAPI.maxUnmaxWindow();
	// change icon
	const icon = maxUnmaxButton.querySelector("i.far");
	const maximized = await window.electronAPI.isWindowMaximized();
	if ( maximized ) {
		icon.classList.remove("fa-square");
		icon.classList.add("fa-clone");
	} else {
		icon.classList.add("fa-square");
		icon.classList.remove("fa-clone");
	}
});

closeButton.addEventListener('click', () => {
	window.electronAPI.closeWindow();
});

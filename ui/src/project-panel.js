import { add_gltf } from './pre-panel'

const gltf_fns= []
const video_fns = []

function parse_dir( paths ) {
    const dir = paths[0];
    const files = paths.slice(1, -1);
    const gltf_list = files.filter(i => i.endsWith('.gltf'));
    const video_list = files.filter(i => i.endsWith('.mp4'));
    return { dir, gltf_list, video_list };
}

function show_project_structure() {
    document.getElementById('no-dataset-hint').style.display = 'none';
    document.getElementById('file-list').style.display = 'flex';
    importBtn.innerText = '更改目录';
}

const importBtn = document.getElementById('import-btn');
importBtn.addEventListener('click', async () => {
    const paths = await window.electronAPI.openFile();
    console.log(paths);
    const { dir, gltf_list, video_list } = parse_dir(paths);
    add_gltf(dir + '/' + gltf_list[0]);
    show_project_structure();
})


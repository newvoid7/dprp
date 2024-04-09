import { add_gltf } from './pre-panel'
import { add_image } from './compare-panel';
import { read_probes } from './sync-panel';

const importBtn = document.getElementById('import-btn');
const refreshBtn = document.getElementById('refresh-btn');
const fileListDiv = document.getElementById('file-list');
const dirName = document.getElementById('dir-name');

var expand = {};
var selected = {};
var fn_list = {};
var dir = '';

function get_last_dir( _dir ) {
    const l = _dir.split('\\')
    return l[l.length - 1];
}

function my_path_join( _dir, fn ) {
    return _dir + '/' + fn;
}

function change_select(class_name, index) {
    const orig_fn = fn_list[class_name][selected[class_name]];
    document.getElementById('file_' + orig_fn).style.backgroundColor = 'transparent';
    const new_fn = fn_list[class_name][index];
    document.getElementById('file_' + new_fn).style.backgroundColor = 'cadetblue';
    if (class_name === 'gltf') {
        add_gltf(my_path_join(dir, new_fn));
        read_probes(
            my_path_join(dir, 'info.json'),
            my_path_join(dir, new_fn)
        );
    } else if (class_name == 'video') {
        add_image(my_path_join(dir, new_fn));
    }
    selected[class_name] = index;
}

function show_file_class(name, display_name, exts, fns) {
    const filtered_fns = fns.filter( fn => {
        const flag = exts.some( e => fn.endsWith(e) );
        return flag;
    });
    expand[name] = true;
    selected[name] = 0;
    fn_list[name] = filtered_fns;
    const top_btn = document.createElement('button');
    top_btn.className = 'filelist-btn';
    top_btn.innerHTML = '-&nbsp;&nbsp;&nbsp;&nbsp;' + display_name;
    // expand
    top_btn.addEventListener('click', async () => {
        if (expand[name]) {
            filtered_fns.forEach(fn => {
                const btn = document.getElementById('file_' + fn);
                btn.style.display = 'none';
            })
            top_btn.innerHTML = '+&nbsp;&nbsp;&nbsp;&nbsp;' + display_name;
            expand[name] = false;
        } else {
            filtered_fns.forEach(fn => {
                const btn = document.getElementById('file_' + fn);
                btn.style.display = 'inline';
            })
            top_btn.innerHTML = '-&nbsp;&nbsp;&nbsp;&nbsp;' + display_name;
            expand[name] = true;
        }
    })
    fileListDiv.appendChild( top_btn );
    filtered_fns.forEach((v, i) => {
        const file_btn = document.createElement('button');
        file_btn.className = 'filelist-btn';
        file_btn.innerHTML = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + v;
        file_btn.id = 'file_' + v;
        file_btn.addEventListener('dblclick', async () => {
            change_select(name, i);
        });
        fileListDiv.appendChild( file_btn );
    })
}

function show_project_structure(fns) {
    document.getElementById('no-dataset-hint').style.display = 'none';
    dirName.style.display = 'flex';
    dirName.innerText = get_last_dir(dir)
    fileListDiv.innerHTML = '';
    Object.keys(expand).forEach( k => delete expand[k] );
    Object.keys(selected).forEach( k => delete selected[k] );
    Object.keys(fn_list).forEach( k => delete fn_list[k] );
    // move the button to the bottom
    importBtn.innerText = '更改目录';
    fileListDiv.style.display = 'flex';
    // fill the file list
    show_file_class('ct', '原始 CT 图像', ['.mhd', '.nii', '.nii.gz'], fns);
    show_file_class('gltf', '三维模型文件', ['.gltf'], fns);
    show_file_class('video', '视频帧文件', ['.png', '.jpg'], fns);
    show_file_class('probe', '渲染探针信息', ['info.json', 'probes.npz'], fns);
    change_select('gltf', 0);
    change_select('video', 0);
}

async function handleImportBtnClick() {
    dir = await window.electronAPI.openFile();
    const fns = await window.electronAPI.listDir(dir);
    refreshBtn.style.display = 'flex';
    show_project_structure(fns);
}

async function handleRefreshBtnClick() {
    const fns = await window.electronAPI.listDir(dir);
    show_project_structure(fns);
}

function self_inc(class_name) {
    var curr = selected[class_name];
    if (curr < fn_list[class_name].length - 1) {
        change_select(class_name, curr + 1);
        return true;
    } else {
        return false;
    }
}

function self_dec() {

}

export {
    fn_list,
    selected,
    change_select,
    handleImportBtnClick,
    handleRefreshBtnClick,
};
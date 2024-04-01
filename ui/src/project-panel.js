import { add_gltf } from './pre-panel'
import { add_image } from './compare-panel';

const importBtn = document.getElementById('import-btn');
const refreshBtn = document.getElementById('refresh-btn');
const fileListDiv = document.getElementById('file-list');
const dirName = document.getElementById('dir-name');

var expand = {};
var selected = {};
var fn_list = {};
var current_dir = '';

function get_last_dir( dir ) {
    const l = dir.split('\\')
    return l[l.length - 1];
}

function my_path_join( dir, fn ) {
    return dir + '/' + fn;
}

function change_select(dir, class_name, index) {
    const orig_fn = fn_list[class_name][selected[class_name]];
    document.getElementById('file_' + orig_fn).style.backgroundColor = 'transparent';
    const new_fn = fn_list[class_name][index];
    document.getElementById('file_' + new_fn).style.backgroundColor = 'cadetblue';
    if (class_name === 'gltf') {
        add_gltf(my_path_join(dir, new_fn));
    } else if (class_name == 'video') {
        add_image(my_path_join(dir, new_fn));
    }
    selected[class_name] = index;
}

function show_file_class(dir, name, display_name, exts, fns) {
    const filtered_fns = fns.filter( fn => {
        const flag = exts.some( e=> fn.endsWith(e) );
        return flag;
    });
    expand[name] = true;
    selected[name] = 0;
    fn_list[name] = filtered_fns;
    const top_btn = document.createElement('button');
    top_btn.className = 'filelist-btn';
    top_btn.innerHTML = '-&nbsp;&nbsp;&nbsp;&nbsp;' + display_name;
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
        file_btn.addEventListener('click', async () => {
            change_select(dir, name, i);
        });
        fileListDiv.appendChild( file_btn );
    })
}

function show_project_structure(dir, fns) {
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
    show_file_class(dir, 'ct', '原始CT图像', ['.mhd', '.nii', '.nii.gz'], fns);
    show_file_class(dir, 'gltf', '三维模型文件', ['.gltf'], fns);
    show_file_class(dir, 'video', '视频帧文件', ['.png', '.jpg'], fns);
    change_select(dir, 'gltf', 0);
    change_select(dir, 'video', 0);
}

importBtn.addEventListener('click', async () => {
    const dir = await window.electronAPI.openFile();
    const fns = await window.electronAPI.listDir(dir);
    current_dir = dir;
    refreshBtn.style.display = 'flex';
    show_project_structure(dir, fns);
})

refreshBtn.addEventListener('click', async () => {
    const fns = await window.electronAPI.listDir(current_dir);
    show_project_structure(current_dir, fns);
})


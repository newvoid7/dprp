const img_ele = document.getElementById('intra-image');


function add_image( path ) {
    img_ele.src = path;
}

export {
    add_image
};
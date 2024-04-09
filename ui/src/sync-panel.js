import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

var probes_json;

const container = document.getElementById( 'sync-renderer' );
const scene = new THREE.Scene();
const camera_film = 1.5;
const camera = new THREE.OrthographicCamera( 
    -camera_film * container.clientWidth / container.clientHeight, camera_film * container.clientWidth / container.clientHeight , 
    camera_film, -camera_film, 
    0.1, 1000
);
const renderer = new THREE.WebGLRenderer( {alpha: true} );
const loader = new GLTFLoader();
const alight = new THREE.AmbientLight( 0xFFFFFF, 5 );
const lights = [
	alight
];

const probe_geo = new THREE.BufferGeometry();
const probe_verts = new Float32Array([
    1,1,0, -1,1,0, -1,-1,0, 1,-1,0, 0,0,1.5, 1,1.2,0, -1,1.2,0, 0,1.7,0
]);
const probe_indes = new Uint16Array([
    0,1,4, 1,2,4, 2,3,4, 3,0,4, 5,6,7
]);
const probe_material = new THREE.MeshBasicMaterial({
    color: 0xf07623,
    wireframe: true
});

probe_geo.setAttribute('position', new THREE.BufferAttribute(probe_verts, 3));
probe_geo.setIndex(new THREE.BufferAttribute(probe_indes, 1));

lights.forEach( l => scene.add(l) );
camera.position.set( 0, -2, 0 );
camera.lookAt( 0, 0, 0 );
camera.up.copy(new THREE.Vector3(0, 0, 1));
renderer.setClearAlpha(0);
container.appendChild( renderer.domElement );

function set_size() {
	renderer.setSize( container.clientWidth, container.clientHeight );
    camera.left = -camera_film * container.clientWidth / container.clientHeight;
    camera.right = camera_film * container.clientWidth / container.clientHeight;
    camera.top = camera_film;
    camera.bottom = -camera_film;
}

async function read_probes( json_path, gltf_path ) {
    const data = await window.electronAPI.readFile(json_path);
    const json_data = await JSON.parse(data);
    probes_json = json_data;
    load_gltf( gltf_path, json_data );
}

function clear_scene() {
	while (scene.children.length > 0) {
		var object = scene.children[0];
        if(object.isMesh) {
            object.geometry.dispose();
            object.material.dispose();
        }
        scene.remove(object);
	}
	lights.forEach( l => scene.add(l) );
}

function create_probes( json_data ) {
    const probe_count = json_data['total'];
    const instance_probe = new THREE.InstancedMesh(
        probe_geo, probe_material, probe_count
    );
    instance_probe.scale.set( 0.6, 0.6, 0.6 );
    for (let i = 0; i < probe_count; ++i) {
        const q = new THREE.Quaternion(
            json_data[i]['quaternion'][1],
            json_data[i]['quaternion'][2],
            json_data[i]['quaternion'][3],
            json_data[i]['quaternion'][0]
        );
        const v = new THREE.Vector3(
            json_data[i]['position'][0],
            json_data[i]['position'][1],
            json_data[i]['position'][2]
        );
        const qmat = new THREE.Matrix4().makeRotationFromQuaternion(q);
        const vmat = new THREE.Matrix4().makeTranslation(v);
        const smat = new THREE.Matrix4().makeScale(
            0.1, 0.1, 0.1
        );
        var matrix = vmat;
        matrix.multiply( qmat );
        matrix.multiply( smat );
        instance_probe.setMatrixAt(i, matrix);
    }
    instance_probe.instanceMatrix.needsUpdate = true;
    scene.add(instance_probe);
}

function load_gltf( model_path, json_data ) {
	loader.load(
		model_path,
		function ( gltf ) {
			// clear
			clear_scene();
			var model = gltf.scene.children[0];
            // only reserve the components that need drawing
            var need_remove = [];
            for (var i = 0; i < model.children.length; ++i) {
                const need_draw = probes_json['draw_mesh'].some( m => m === i);
                if (!need_draw) {
                    need_remove.push(model.children[i]);
                }
            }
            need_remove.forEach( m => {
                if(m.isMesh) {
                    m.geometry.dispose();
                    m.material.dispose();
                }
                model.remove(m);
            });
			// normalize
			var bbx = new THREE.Box3().setFromObject( model );
			var length = bbx.clone().max.sub( bbx.min );
			var l = Math.max( length.x, length.y, length.z );
			model.position.set( -0.5, -0.5, -0.5 );
			model.scale.set( 1/l, 1/l, 1/l );
			scene.add( model );
            create_probes( json_data );
		},
		function ( xhr ) {
			// console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
		},
		function ( error ) {
			console.log( error );
		}
	);
}

function animate() {
	requestAnimationFrame( animate );
    scene.rotation.z += 0.005;
	renderer.render( scene, camera );
	set_size();
}

animate();

export {
    read_probes
};
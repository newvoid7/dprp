import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const container = document.getElementById( 'pre-renderer' );
const hint = document.getElementById( 'pre-hint' );

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, container.clientWidth / container.clientHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer( {alpha: true} );
const loader = new GLTFLoader();
const alight = new THREE.AmbientLight( 0xFFFFFF, 1 );
const dlight = new THREE.DirectionalLight( 0xFFFFFF, 5 );

const lights = [
	alight, dlight
];

container.appendChild( renderer.domElement );

dlight.lookAt( 0, 0, 0 );
dlight.position.set( 10, 10, 10 );
lights.forEach( l => scene.add(l) );
camera.position.set( 0, 1, 0 );
camera.lookAt( 0, 0, 0 );
camera.up.copy(new THREE.Vector3(0, 0, 1));
renderer.setClearAlpha(0);

function vector_to_str( v ) {
	var ret = '( ';
	if ( v.isVector3 ) {
		ret += v.x.toFixed(2) + ', ' + v.y.toFixed(2) + ', ' + v.z.toFixed(2);
	} else if ( v.isQuaternion ) {
		ret += v._x.toFixed(2) + ', ' + v._y.toFixed(2) + ', ' + v._z.toFixed(2) + ', ' + v._w.toFixed(2);
	}
	ret += ' )';
	return ret;
}

function set_size() {
	renderer.setSize( container.clientWidth, container.clientHeight );
	camera.aspect = container.clientWidth / container.clientHeight;
	camera.updateProjectionMatrix();
}

function set_hint() {
	var hint_str = 'Camera <br> position: ' + vector_to_str(camera.position);
	hint_str += '<br> direction: ' + vector_to_str(camera.quaternion);
	hint_str += '<br> fov: ' + camera.fov;
	var azimuth = Math.atan2( camera.position.x, camera.position.y );
	var zenith = Math.acos( camera.postion.z / camera.position.length() );
	hint_str += '<br> azimuth: ' + azimuth;
	hint_str += '<br> zenith: ' + zenith;
	hint.innerHTML = hint_str;
}

function clear_scene() {
	while (scene.children.length > 0) {
		var object = scene.children[0];
        if(object.isMesh) {
            // 清除网格的geometry和material
            object.geometry.dispose();
            object.material.dispose();
        }
        scene.remove(object);
	}
	lights.forEach( l => scene.add(l) );
}

function add_gltf( path ) {
	loader.load(
		path,
		function ( gltf ) {
			// clear
			clear_scene();
			var model = gltf.scene;
			// normalize
			var bbx = new THREE.Box3().setFromObject( model );
			var length = bbx.clone().max.sub( bbx.min );
			var l = Math.max( length.x, length.y, length.z );
			model.position.set( -0.5, -0.5, -0.5 );
			model.scale.set( 1/l, 1/l, 1/l );
			scene.add( model );
			hint.style.visibility = 'visible';
		},
		function ( xhr ) {
			console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
		},
		function ( error ) {
			console.log( error );
		}
	);
}

const controls = new OrbitControls( camera, renderer.domElement );

function animate() {
	requestAnimationFrame( animate );
	renderer.render( scene, camera );
	set_size();
	set_hint();
}

animate();

export {
    add_gltf
};
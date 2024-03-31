import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const container = document.getElementById( 'pre-renderer' );

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, container.clientWidth / container.clientHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer( {alpha: true} );
const loader = new GLTFLoader();
const alight = new THREE.AmbientLight( 0xFFFFFF, 1 );
const dlight = new THREE.DirectionalLight( 0xFFFFFF, 5 );

container.appendChild( renderer.domElement );

dlight.lookAt( 0, 0, 0 );
dlight.position.set( 10, 10, 10 );
scene.add( alight );
scene.add( dlight );
camera.position.set( 0, 0, 1.2 );
camera.lookAt( 0, 0, 0 );
camera.up.copy(new THREE.Vector3(0, 1, 0));
renderer.setClearAlpha(0);

function set_size() {
	renderer.setSize( container.clientWidth, container.clientHeight );
	camera.aspect = container.clientWidth / container.clientHeight;
	camera.updateProjectionMatrix();
}

function add_gltf( path ) {
	loader.load(
		path,
		function ( gltf ) {
			var model = gltf.scene;
			// normalize
			var bbx = new THREE.Box3().setFromObject( model );
			var length = bbx.clone().max.sub( bbx.min );
			var l = Math.max( length.x, length.y, length.z );
			model.position.set( -0.5, -0.5, -0.5 );
			model.scale.set( 1/l, 1/l, 1/l );
			scene.add( model );
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
}

animate();

export {
    add_gltf
};
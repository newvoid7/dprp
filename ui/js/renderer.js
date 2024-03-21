import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 
	75, 
	window.innerWidth / window.innerHeight, 
	0.1, 
	1000 
);
const renderer = new THREE.WebGLRenderer();
const loader = new GLTFLoader();
const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry, material );

document.getElementById('webglviewer').appendChild( renderer.domElement )

camera.position.set( 0, 0, 5 );
camera.lookAt( 0, 0, 0 );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
scene.add( cube );
loader.load( 
	'mesh.gltf', 
	// called when the resource is loaded
	function ( gltf ) {
		scene.add( gltf.scene );
	},
	// called while loading is progressing
	function ( xhr ) {
		console.log( ( xhr.loaded / xhr.total * 100 ) + '% loaded' );
	},
	// called when loading has errors
	function ( error ) {
		console.log( 'An error happened' + error );
	}
);
animate();

function animate() {
	requestAnimationFrame( animate );
	renderer.render( scene, camera );
}
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const path = require('node:path')

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 
	75, 
	window.innerWidth / window.innerHeight, 
	0.1, 
	1000 
);
const renderer = new THREE.WebGLRenderer();
const loader = new GLTFLoader();

document.getElementById('webglviewer').appendChild( renderer.domElement )

camera.position.set( 0, 0, 5 );
camera.lookAt( 0, 0, 0 );
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );

loader.load( 
	path.join(__dirname, 'mesh.gltf'), 
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
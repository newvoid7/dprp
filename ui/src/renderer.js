/**
 * This file will automatically be loaded by vite and run in the "renderer" context.
 * To learn more about the differences between the "main" and the "renderer" context in
 * Electron, visit:
 *
 * https://electronjs.org/docs/tutorial/application-architecture#main-and-renderer-processes
 *
 * By default, Node.js integration in this file is disabled. When enabling Node.js integration
 * in a renderer process, please be aware of potential security implications. You can read
 * more about security risks here:
 *
 * https://electronjs.org/docs/tutorial/security
 *
 * To enable Node.js integration in this file, open up `main.js` and enable the `nodeIntegration`
 * flag:
 *
 * ```
 *  // Create the browser window.
 *  mainWindow = new BrowserWindow({
 *    width: 800,
 *    height: 600,
 *    webPreferences: {
 *      nodeIntegration: true
 *    }
 *  });
 * ```
 */

import './index.css';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
const loader = new GLTFLoader();
const alight = new THREE.AmbientLight( 0xFFFFFF, 1 );
const dlight = new THREE.DirectionalLight( 0xFFFFFF, 10 );

document.getElementById( '3d-renderer' ).appendChild( renderer.domElement );

dlight.lookAt( 0, 0, 0 );
dlight.position.set( 10, 10, 10 );
scene.background = new THREE.Color( '#505050' );
scene.add( alight );
scene.add( dlight );
camera.position.set( 0, 0, 3 );
camera.lookAt( 0, 0, 0 );
renderer.setSize( 0.5 * window.innerWidth, 0.5 * window.innerHeight );

document.getElementById('file-selector').addEventListener('change', 
	function( event ) {
		var path = event.target.files[0].path;
		console.log( path );
		add_gltf( path );
	}
)

function add_gltf( path ) {
	loader.load(
		path,
		function ( gltf ) {
			var model = gltf.scene;
			// normalize
			var bbx = new THREE.Box3().setFromObject( model );
			var c = bbx.clone().max.add( bbx.min ).multiplyScalar( 0.5 );
			var length = bbx.clone().max.sub( bbx.min );
			var l = Math.max( length.x, length.y, length.z );
			model.position.set( -c.x, -c.y, -c.z );
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

function animate() {
	requestAnimationFrame( animate );
	renderer.render( scene, camera );
}

animate();

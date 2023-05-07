var gl;
var canvas;
var shaderProgram;
var vertexPositionBuffer;

var days = 0;

// Create a place to store sphere geometry
var sphereVertexPositionBuffer;

//Create a place to store normals for shading
var sphereVertexNormalBuffer;

// View parameters
var eyePt = vec3.fromValues(0.0, 0.0, 4.0);
var viewDir = vec3.fromValues(0.0, 0.0, -1.0);
var up = vec3.fromValues(0.0, 1.0, 0.0);
var viewPt = vec3.fromValues(0.0, 0.0, 0.0);

// Create the normal
var nMatrix = mat3.create();

// Create ModelView matrix
var mvMatrix = mat4.create();

//Create Projection matrix
var pMatrix = mat4.create();

var mvMatrixStack = [];

// Shiness
var shiny = 100;

// List of particles
var particles = [];

// Initialize particles list
addParticles();

//Color conversion  helper functions
function hexToR(h) { return parseInt((cutHex(h)).substring(0, 2), 16) }
function hexToG(h) { return parseInt((cutHex(h)).substring(2, 4), 16) }
function hexToB(h) { return parseInt((cutHex(h)).substring(4, 6), 16) }
function cutHex(h) { return (h.charAt(0) == "#") ? h.substring(1, 7) : h }


//Populates buffers with data for spheres
function setupSphereBuffers() {

  var sphereSoup = [];
  var sphereNormals = [];
  var numT = sphereFromSubdivision(6, sphereSoup, sphereNormals);

  sphereVertexPositionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexPositionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sphereSoup), gl.DYNAMIC_DRAW);
  sphereVertexPositionBuffer.itemSize = 3;
  sphereVertexPositionBuffer.numItems = numT * 3;

  // Specify normals to be able to do lighting calculations
  sphereVertexNormalBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexNormalBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sphereNormals),
    gl.DYNAMIC_DRAW);
  sphereVertexNormalBuffer.itemSize = 3;
  sphereVertexNormalBuffer.numItems = numT * 3;
}

// Set up timer for periodic reset
let minResetTime = 3000; // 3 seconds
let maxResetTime = 15000; // 15 seconds
let nextResetTime = Date.now() + Math.random() * (maxResetTime - minResetTime) + minResetTime;

//Draws a sphere from the sphere buffer

function drawSphere() {
  gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexPositionBuffer);
  gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, sphereVertexPositionBuffer.itemSize,
    gl.FLOAT, false, 0, 0);

  // Bind normal buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, sphereVertexNormalBuffer);
  gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute,
    sphereVertexNormalBuffer.itemSize,
    gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLES, 0, sphereVertexPositionBuffer.numItems);
}

//Sends Modelview matrix to shader
function uploadModelViewMatrixToShader() {
  gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, mvMatrix);
}

// Sends projection matrix to shader
 
function uploadProjectionMatrixToShader() {
  gl.uniformMatrix4fv(shaderProgram.pMatrixUniform,
    false, pMatrix);
}

// Generates and sends the normal matrix to the shader
function uploadNormalMatrixToShader() {
  mat3.fromMat4(nMatrix, mvMatrix);
  mat3.transpose(nMatrix, nMatrix);
  mat3.invert(nMatrix, nMatrix);
  gl.uniformMatrix3fv(shaderProgram.nMatrixUniform, false, nMatrix);
}

// Pushes matrix onto modelview matrix stack
function mvPushMatrix() {
  var copy = mat4.clone(mvMatrix);
  mvMatrixStack.push(copy);
}

// Pops matrix off of modelview matrix stack
function mvPopMatrix() {
  if (mvMatrixStack.length == 0) {
    throw "Invalid popMatrix!";
  }
  mvMatrix = mvMatrixStack.pop();
}

 //Sends projection/modelview matrices to shader
function setMatrixUniforms() {
  uploadModelViewMatrixToShader();
  uploadNormalMatrixToShader();
  uploadProjectionMatrixToShader();
}

//----------------------------------------------------------------------------------
/**
 * Translates degrees to radians
 * @param {Number} degrees Degree input to function
 * @return {Number} The radians that correspond to the degree input
 */
function degToRad(degrees) {
  return degrees * Math.PI / 180;
}

//----------------------------------------------------------------------------------
/**
 * Creates a context for WebGL
 * @param {element} canvas WebGL canvas
 * @return {Object} WebGL context
 */
function createGLContext(canvas) {
  var names = ["webgl2", "experimental-webgl2"];
  var context = null;
  for (var i = 0; i < names.length; i++) {
    try {
      context = canvas.getContext(names[i]);
    } catch (e) { }
    if (context) {
      break;
    }
  }
  if (context) {
    context.viewportWidth = canvas.width;
    context.viewportHeight = canvas.height;
  } else {
    alert("Failed to create WebGL context!");
  }
  return context;
}

const vertexShaderSource = `#version 300 es
  in vec3 aVertexNormal;
  in vec3 aVertexPosition;

  uniform mat4 uMVMatrix;
  uniform mat4 uPMatrix;
  uniform mat3 uNMatrix;

  out vec3 vPosition;
  out vec3 vNormal;

  void main(void) {
    vec4 vertexPositionEye4 = uMVMatrix * vec4(aVertexPosition, 1.0);
    vPosition = vertexPositionEye4.xyz / vertexPositionEye4.w;

    vNormal = normalize(uNMatrix * aVertexNormal);

    gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
  }`
  
const fragmentShaderSource = `#version 300 es
  precision mediump float;

  uniform vec3 uLightPosition;
  uniform vec3 uAmbientLightColor;
  uniform vec3 uDiffuseLightColor;
  uniform vec3 uSpecularLightColor;
  uniform vec3 uAmbientMaterialColor;
  uniform vec3 uDiffuseMaterialColor;
  uniform vec3 uSpecularMaterialColor;

  uniform float uShininess;

  in vec3 vPosition;
  in vec3 vNormal;

  out vec4 fragColor;

  void main(void) {
    vec3 vectorToLightSource = normalize(uLightPosition - vPosition);

    float diffuseLightWeightning = max(dot(vNormal, vectorToLightSource), 0.0);

    vec3 reflectionVector = normalize(reflect(-vectorToLightSource, vNormal));

    vec3 viewVectorEye = -normalize(vPosition);

    float rdotv = max(dot(reflectionVector, viewVectorEye), 0.0);
	float specularLightWeightning = pow(rdotv, uShininess);

    fragColor = vec4(((uAmbientLightColor * uAmbientMaterialColor)
        + (uDiffuseLightColor * uDiffuseMaterialColor) * diffuseLightWeightning
        + (uSpecularLightColor * uSpecularMaterialColor) * specularLightWeightning), 1.0);
  }`
  
let lastTime = 0;
let frameCount = 0;

function updateFPS(time) {
  frameCount++;

  // Calculate the elapsed time since the last frame
  let elapsedTime = time - lastTime;

  if (elapsedTime >= 1000) { // Update the FPS display every 1000ms (1 second)
    let fps = (frameCount / elapsedTime) * 1000;
    //document.getElementById('fps').innerHTML = `FPS: ${fps.toFixed(2)}`;
	document.querySelector('#fps').innerHTML = `FPS: ${fps.toFixed(2)}`;
    frameCount = 0;
    lastTime = time;
  }
}
/**
 * Loads Shaders
 * @param {string} id ID string for shader to load. Either vertex shader/fragment shader
 */
function loadShader(shaderSource, shaderType) {
  var shader = gl.createShader(shaderType);
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    alert(gl.getShaderInfoLog(shader));
    return null;
  }
  return shader;
}

//Setup the fragment and vertex shaders
 
 function compileAndLink(vertexShaderSource, fragmentShaderSource) {
  const vertexShader = loadShader(vertexShaderSource, gl.VERTEX_SHADER);
  const fragmentShader = loadShader(fragmentShaderSource, gl.FRAGMENT_SHADER);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Failed to link shader program:', gl.getProgramInfoLog(program));
    return null;
  }

  return program;
}
 
function setupShaders(vshaderSource, fshaderSource) {
  shaderProgram = compileAndLink(vshaderSource, fshaderSource);

  if (!shaderProgram) {
    alert("Failed to setup shaders");
    return;
  }

  gl.useProgram(shaderProgram);
  shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
  gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
  shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
  gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);
  shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
  shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
  shaderProgram.nMatrixUniform = gl.getUniformLocation(shaderProgram, "uNMatrix");
  shaderProgram.uniformLightPositionLoc = gl.getUniformLocation(shaderProgram, "uLightPosition");
  shaderProgram.uniformAmbientLightColorLoc = gl.getUniformLocation(shaderProgram, "uAmbientLightColor");
  shaderProgram.uniformDiffuseLightColorLoc = gl.getUniformLocation(shaderProgram, "uDiffuseLightColor");
  shaderProgram.uniformSpecularLightColorLoc = gl.getUniformLocation(shaderProgram, "uSpecularLightColor");
  shaderProgram.uniformDiffuseMaterialColor = gl.getUniformLocation(shaderProgram, "uDiffuseMaterialColor");
  shaderProgram.uniformAmbientMaterialColor = gl.getUniformLocation(shaderProgram, "uAmbientMaterialColor");
  shaderProgram.uniformSpecularMaterialColor = gl.getUniformLocation(shaderProgram, "uSpecularMaterialColor");

  shaderProgram.uniformShininess = gl.getUniformLocation(shaderProgram, "uShininess");
}
/**
 * Sends material information to the shader
 * @param {Float32Array} a diffuse material color
 * @param {Float32Array} a ambient material color
 * @param {Float32Array} a specular material color 
 * @param {Float32} the shininess exponent for Phong illumination
 */
function uploadMaterialToShader(dcolor, acolor, scolor, shiny) {
  gl.uniform3fv(shaderProgram.uniformDiffuseMaterialColor, dcolor);
  gl.uniform3fv(shaderProgram.uniformAmbientMaterialColor, acolor);
  gl.uniform3fv(shaderProgram.uniformSpecularMaterialColor, scolor);
  gl.uniform1f(shaderProgram.uniformShininess, shiny);
}
//-------------------------------------------------------------------------
/**
 * Sends light information to the shader
 * @param {Float32Array} loc Location of light source
 * @param {Float32Array} a Ambient light strength
 * @param {Float32Array} d Diffuse light strength
 * @param {Float32Array} s Specular light strength
 */
function uploadLightsToShader(loc, a, d, s) {
  gl.uniform3fv(shaderProgram.uniformLightPositionLoc, loc);
  gl.uniform3fv(shaderProgram.uniformAmbientLightColorLoc, a);
  gl.uniform3fv(shaderProgram.uniformDiffuseLightColorLoc, d);
  gl.uniform3fv(shaderProgram.uniformSpecularLightColorLoc, s);
}
//Populate buffers with data

function setupBuffers() {
  setupSphereBuffers();
}

//Draw call that applies matrix transformations to model and draws model in frame

function draw() {
  gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // We'll use perspective 
  mat4.perspective(pMatrix, degToRad(45), gl.viewportWidth / gl.viewportHeight, 0.1, 200.0);

  // We want to look down -z, so create a lookat point in that direction    
  vec3.add(viewPt, eyePt, viewDir);
  // Then generate the lookat matrix and initialize the MV matrix to that view
  mat4.lookAt(mvMatrix, eyePt, viewPt, up);

  for (var i = 0; i < particles.length; i++) {
    var p = particles[i];
    mvPushMatrix();
    mat4.translate(mvMatrix, mvMatrix, p.p);
    mat4.scale(mvMatrix, mvMatrix, [p.r, p.r, p.r]);

    //Get material color
    R = p.R;
    G = p.G;
    B = p.B;

    uploadLightsToShader([0, 0, 0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    uploadMaterialToShader([R, G, B], [R, G, B], [1.0, 1.0, 1.0], shiny);
    setMatrixUniforms();
    drawSphere();
    mvPopMatrix();

  }
}
//Animation to be called from tick. Updates globals and performs animation for each tick.

function setGouraudShader() {
  setupShaders("shader-gouraud-phong-vs", "shader-gouraud-phong-fs");
}

// Startup function called from html code to start program.
function startup() {
  canvas = document.getElementById("invisibleCube");
  gl = createGLContext(canvas);
  setupShaders(vertexShaderSource,fragmentShaderSource);
  setupBuffers();
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.enable(gl.DEPTH_TEST);
  tick();
}

// Tick called for every animation frame.
function tick(milliseconds) {
  requestAnimationFrame(tick);
  let currentTime = Date.now();
if (currentTime >= nextResetTime) {
resetParticles();
addParticles();
nextResetTime = currentTime + Math.random() * (maxResetTime - minResetTime) + minResetTime;
}

 updateParticles();
  draw();
  detectCollisions(); 
  updateFPS(milliseconds);
}

//Update particle positions, velocities and accelerations
function updateParticles() {
  updateParameters();
  for (var i = 0; i < particles.length; i++) {
    particles[i].updateVelocity();
    particles[i].updatePosition();
    particles[i].updateAcceleration();
  }
}

//Add particles
function addParticles() {
  // Ensure no particles are added with overlapping positions
  let attempts = 10000; // Limit the number of attempts to find a non-overlapping position
  let minRadius = 0.1;
  let maxRadius = minRadius * 3; // At least a 3-fold radius difference
  let massFactor = 100; // At least a 10-fold mass difference

  for (let i = 0; i < 50; i++) {
    let radius = Math.random() * (maxRadius - minRadius) + minRadius; // Assign random radius between minRadius and maxRadius
    let newParticle = {
      p: vec3.fromValues(Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1),
      r: radius,
      m: Math.pow(radius, 3) * massFactor, // Mass proportional to volume (radius³) with a 10-fold mass difference
      R: Math.random(),
      G: Math.random(),
      B: Math.random()
    };

    let isOverlapping = false;
    for (let j = 0; j < particles.length; j++) {
      let distance = vec3.distance(particles[j].p, newParticle.p);
      if (distance <= particles[j].r + newParticle.r) {
        isOverlapping = true;
        break;
      }
    }
    
    if (!isOverlapping) {
      particles.push(new Particle(newParticle)); // Pass newParticle as an argument
    } else {
      attempts--;
      if (attempts <= 0) {
        console.warn("Failed to add all particles without overlapping.");
        break;
      }
      i--; // Retry the same iteration
    }
  }
}
function detectCollisions() {
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      let distance = vec3.distance(particles[i].p, particles[j].p);
      let sumRadii = particles[i].r + particles[j].r;

      if (distance <= sumRadii) {
        // Spheres are colliding
        let collisionVector = vec3.create();
        vec3.sub(collisionVector, particles[i].p, particles[j].p);
        vec3.normalize(collisionVector, collisionVector);

        let overlap = sumRadii - distance;
        vec3.scale(collisionVector, collisionVector, overlap / 2);

        // Move the spheres apart based on the collision vector
        vec3.add(particles[i].p, particles[i].p, collisionVector);
        vec3.sub(particles[j].p, particles[j].p, collisionVector);
      }
    }
  }
}

 //Reset particles
function resetParticles() {
  particles = [];
}

 //Burst particles
function burstParticles() {
  for (var i = 0; i < particles.length; i++) {
    vec3.random(particles[i].v, 2);
  }
}
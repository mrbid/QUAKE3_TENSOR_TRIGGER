# QUAKE3_TENSOR_TRIGGER
A machine learning trigger bot for Quake III Arena &amp; Quake Live.

### All you have to do is force player models to the aqua blue bones model as that is the only character model I trained this dataset on.

### prerequisites 
```
sudo apt install clang xterm python3 python3-pip libx11-dev libxdo-dev libxdo3 libespeak1 libespeak-dev espeak
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install numpy
sudo python3 -m pip install tensorflow-cpu
sudo python3 -m pip install tf2onnx
sudo python3 -m pip install onnxruntime
```

### exec
```
./RUN_CNN.sh
or
./RUN_FNN.sh
```

This is cool but [this one is better for low end hardware](https://gist.github.com/mrbid/37996a90792f26bd02787fd4ab8b1bad).

<details>
    <summary>Q3 Config for low end hardware</summary>
cg_oldRail "1"<br>
cg_noProjectileTrail "1"<br>
cg_forceModel "1"<br>
cg_railTrailTime "0"<br>
cg_drawFPS "1"<br>
cg_draw2D "1"<br>
cg_gibs "0"<br>
cg_fov "150"<br>
cg_zoomfov "90"<br>
cg_drawGun "1"<br>
cg_brassTime "0"<br>
cg_drawCrosshair "0"<br>
cg_drawCrosshairNames "1"<br>
cg_marks "0"<br>
cg_centertime "0"<br>
xp_noParticles "1"<br>
xp_noShotgunTrail "1"<br>
xp_noMip "2047"<br>
xp_ambient "1"<br>
xp_modelJump "0"<br>
xp_corpse "3"<br>
xp_improvePrediction "1"<br>
cm_playerCurveClip "1"<br>
com_maxfps "250"<br>
com_blood "0"<br>
cg_autoswitch "0"<br>
model "bones/default"<br>
headmodel "bones/default"<br>
team_model "bones/default"<br>
team_headmodel "bones/default"<br>
color1 "6"<br>
color2 "5"<br>
r_picmip "16"<br>
r_overBrightBits "1"<br>
r_simpleMipMaps "1"<br>
r_vertexLight "0"<br>
cg_shadows "0"<br>
</details>

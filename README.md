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

This is cool but [this one is simpler and better](https://gist.github.com/mrbid/37996a90792f26bd02787fd4ab8b1bad).

<details>
    <summary>Q3 Config for low end hardware</summary>
cg_oldRail "1"
cg_noProjectileTrail "1"
cg_forceModel "1"
cg_railTrailTime "0"
cg_drawFPS "1"
cg_draw2D "1"
cg_gibs "0"
cg_fov "150"
cg_zoomfov "90"
cg_drawGun "1"
cg_brassTime "0"
cg_drawCrosshair "0"
cg_drawCrosshairNames "1"
cg_marks "0"
cg_centertime "0"
xp_noParticles "1"
xp_noShotgunTrail "1"
xp_noMip "2047"
xp_ambient "1"
xp_modelJump "0"
xp_corpse "3"
xp_improvePrediction "1"
cm_playerCurveClip "1"
com_maxfps "250"
com_blood "0"
cg_autoswitch "0"
model "bones/default"
headmodel "bones/default"
team_model "bones/default"
team_headmodel "bones/default"
color1 "6"
color2 "5"
cg_predictItems "1"
r_picmip "16"
r_overBrightBits "1"
r_simpleMipMaps "1"
r_vertexLight "0"
cg_shadows "0"
</details>

/*
--------------------------------------------------
    James William Fletcher (github.com/mrbid)
--------------------------------------------------
        DECEMBER 2022
        ~~~~~~~~~~~~~

    Force models to aqua blue bones!

    QuakeLive Settings:
    /r_picmip 16
    /cg_shadows 0
    /com_maxfps 333
    /cg_drawfps 1
    /cg_fov 130
    /cg_railTrailTime 0
    
    Prereq:
    sudo apt install xdo wmctrl espeak xdotool libxdo-dev libxdo3
    sudo apt install libxdo-dev libxdo3 libespeak1 libespeak-dev espeak

    ONNX (make fps go zoom):
    Opsets: https://onnxruntime.ai/docs/reference/compatibility.html
    sudo python3 -m pip install tf2onnx
    sudo python3 -m pip install onnxruntime

    Compile:
    clang qaim.c -Ofast -mfma -lX11 -lxdo -lespeak -lm -o aim

    Launch:
    while true; do ./aim; done
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <X11/Xutil.h>
#include <sys/stat.h>

#include <xdo.h>
#include <espeak/speak_lib.h>

#pragma GCC diagnostic ignored "-Wgnu-folding-constant"
#pragma GCC diagnostic ignored "-Wunused-result"

#define SCAN_DELAY 1000
#define ACTIVATION_SENITIVITY 0.97f
#define REPEAT_ACTIVATION 0

#define uint unsigned int
#define SCAN_WIDTH 28
#define SCAN_HEIGHT 28

const uint sw = SCAN_WIDTH;
const uint sh = SCAN_HEIGHT;
const uint sw2 = sw/2;
const uint sh2 = sh/2;
const uint slc = sw*sh;
const uint slall = slc*3;

uint sps = 0; // for SPS

Display *d;
int si;
Window twin;
unsigned int x=0, y=0;
unsigned int tc = 0;

unsigned char rgbbytes[slall] = {0};
char targets_dir[256];


/***************************************************
   ~~ Neural Network Forward-Pass
*/
void processScanArea(Window w);
void writePPM(const char* file, const unsigned char* data);

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

float processModel()
{
    // prevent old outputs returning
    static uint64_t stale = 0;
    if(microtime() - stale > 60000)
    {
        remove("/dev/shm/pred_r.dat");
        stale = microtime();
        return 0.f;
    }

    // load last result
    float ret = 0.f;
    FILE* f = fopen("/dev/shm/pred_r.dat", "rb");
    if(f != NULL)
    {
        if(fread(&ret, 1, sizeof(float), f) != sizeof(float))
        {
            fclose(f);
            return 0.f;
        }
        fclose(f);
        remove("/dev/shm/pred_r.dat");

        // grab a new sceen buffer
        processScanArea(twin);
        
        // write sceen buffer / nn input to file
        writePPM("/dev/shm/pred_input.dat", &rgbbytes[0]);
    }
    else
    {
        if(access("/dev/shm/pred_input.dat", F_OK) != 0)
        {
            // grab a new sceen buffer
            processScanArea(twin);
            
            // write sceen buffer / nn input to file
            writePPM("/dev/shm/pred_input.dat", &rgbbytes[0]);
        }
    }

    // return
    stale = microtime();
    return ret;
}

/***************************************************
   ~~ Utils
*/
//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
//https://stackoverflow.com/questions/18281412/check-keypress-in-c-on-linux/52801588
int key_is_pressed(KeySym ks)
{
    char keys_return[32];
    XQueryKeymap(d, keys_return);
    KeyCode kc2 = XKeysymToKeycode(d, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    return isPressed;
}

unsigned int espeak_fail = 0;
void speakS(const char* text)
{
    if(espeak_fail == 1)
    {
        char s[256];
        sprintf(s, "/usr/bin/espeak \"%s\"", text);
        system(s);
        usleep(33000);
    }
    else
    {
        espeak_Synth(text, strlen(text), 0, 0, 0, espeakCHARS_AUTO, NULL, NULL);
    }
}

void writePPM(const char* file, const unsigned char* data)
{
    FILE* f = fopen(file, "wb");
    if(f != NULL)
    {
        fprintf(f, "P6 28 28 255 ");
        fwrite(data, 1, slall, f);
        fclose(f);
    }
}

/***************************************************
   ~~ X11 Utils
*/

Window getWindow(Display* d, const int si) // gets child window mouse is over
{
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }
    return event.xbutton.window;
}

Window findWindow(Display *d, Window current, char const *needle)
{
    Window ret = 0, root, parent, *children;
    unsigned cc;
    char *name = NULL;

    if(current == 0)
        current = XDefaultRootWindow(d);

    if(XFetchName(d, current, &name) > 0)
    {
        int r = strcmp(needle, name);
        XFree(name);
        if(r == 0)
            return current;
    }

    if(XQueryTree(d, current, &root, &parent, &children, &cc) != 0)
    {
        for(unsigned int i = 0; i < cc; ++i)
        {
            Window win = findWindow(d, children[i], needle);

            if(win != 0)
            {
                ret = win;
                break;
            }
        }
        XFree(children);
    }
    return ret;
}

Window getNextChild(Display* d, Window current)
{
    unsigned int cc = 0;
    Window root, parent, *children;
    if(XQueryTree(d, current, &root, &parent, &children, &cc) == 0)
        return current;
    const Window rw = children[0];
    XFree(children);
    //printf("%lX\n", children[i]);
    return rw;
}

void saveSample(Window w, const char* name)
{
    // save to file
    writePPM(name, &rgbbytes[0]);
}

void processScanArea(Window w)
{
    // get image block
    XImage *img = XGetImage(d, w, x-sw2, y-sh2, sw, sh, AllPlanes, XYPixmap);
    if(img == NULL)
        return;

    // extract colour information
    int i = 0;
    for(int y = 0; y < sh; y++)
    {
        for(int x = 0; x < sw; x++)
        {
            const unsigned long pixel = XGetPixel(img, x, y);
            const unsigned char sr = (pixel & img->red_mask) >> 16;
            const unsigned char sg = (pixel & img->green_mask) >> 8;
            const unsigned char sb = pixel & img->blue_mask;

            // 0-1 norm
            rgbbytes[i]   = (unsigned char)sr;
            rgbbytes[++i] = (unsigned char)sg;
            rgbbytes[++i] = (unsigned char)sb;
            i++;
        }
    }

    // free image block
    XDestroyImage(img);

    // increment SPS
    sps++;
}

/***************************************************
   ~~ Console Utils
*/

// int gre()
// {
//     int r = 0;
//     while(r == 0 || r == 15 || r == 16 || r == 189)
//     {
//         r = (rand()%229)+1;
//     }
//     return r;
// }
// void random_printf(const char* text)
// {
//     const unsigned int len = strlen(text);
//     for(unsigned int i = 0; i < len; i++)
//     {
//         printf("\e[38;5;%im", gre());
//         printf("%c", text[i]);
//     }
//     printf("\e[38;5;123m");
// }

void rainbow_printf(const char* text)
{
    static unsigned char base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    base_clr += 3;

    unsigned char clr = base_clr;
    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
    {
        clr++;
        printf("\e[38;5;%im", clr);
        printf("%c", text[i]);
    }
    printf("\e[38;5;123m");
}

void rainbow_line_printf(const char* text)
{
    static unsigned char base_clr = 0;
    if(base_clr == 0)
        base_clr = (rand()%125)+55;
    
    printf("\e[38;5;%im", base_clr);
    base_clr++;
    if(base_clr >= 230)
        base_clr = (rand()%125)+55;

    const unsigned int len = strlen(text);
    for(unsigned int i = 0; i < len; i++)
        printf("%c", text[i]);
    printf("\e[38;5;123m");
}

/***************************************************
   ~~ Program Entry Point
*/
int main()
{
    srand(time(0));

    // wipe old data
    remove("/dev/shm/pred_r.dat");
    remove("/dev/shm/pred_input.dat");

    if(espeak_Initialize(AUDIO_OUTPUT_SYNCH_PLAYBACK, 0, 0, 0) < 0)
        espeak_fail = 1;

    system("clear");
    rainbow_printf("James William Fletcher (github.com/mrbid)\n\n");
    rainbow_printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    rainbow_printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    rainbow_printf("P = Toggle crosshair\n\n");
    rainbow_printf("L = Toggle sample capture\n");
    rainbow_printf("G = Get activation for reticule area.\n");
    rainbow_printf("H = Get scans per second.\n");
    rainbow_printf("\nDisable the game crosshair and use the one provided by this bot, or if your monitor provides a crosshair use that.\n\n");

    xdo_t* xdo;
    XColor c[9];
    GC gc = 0;
    unsigned int enable = 0;
    unsigned int offset = 3;
    unsigned int crosshair = 1;
    unsigned int sample_capture = 0;
    unsigned int hotkeys = 1;
    unsigned int draw_sa = 0;
    time_t ct = time(0);

    // open display 0
    d = XOpenDisplay(":0");
    if(d == NULL)
    {
        printf("Failed to open display\n");
        return 0;
    }

    // get default screen
    si = XDefaultScreen(d);

    //xdo
    xdo = xdo_new(":0");

    // set console title
    // system("xdotool getactivewindow set_window --name \"QuakeLive Autoshoot\"");
    // Window awin;
    // xdo_get_active_window(xdo, &awin);
    // xdo_set_window_property(xdo, awin, "WM_NAME", "QuakeLive Autoshoot");

    // get graphics context
    gc = DefaultGC(d, si);

    // find bottom window
    twin = findWindow(d, 0, "Quake Live");
    if(twin != 0)
    {
        printf("QL Win: 0x%lX\n", twin);
        twin = getNextChild(d, twin);
        printf("QL Win Child: 0x%lX\n\n", twin);
    }
    
    while(1)
    {
        // loop every 1 ms (1,000 microsecond = 1 millisecond)
        usleep(SCAN_DELAY);

        // inputs
        if(key_is_pressed(XK_Control_L) && key_is_pressed(XK_Alt_L))
        {
            if(enable == 0)
            {                
                // get window
                //xdo_get_active_window(xdo, &twin);
                twin = findWindow(d, 0, "Quake Live");
                if(twin != 0)
                    twin = getNextChild(d, twin);
                else
                    twin = getWindow(d, si);

                // get center window point (x & y)
                XWindowAttributes attr;
                XGetWindowAttributes(d, twin, &attr);
                x = attr.width/2;
                y = attr.height/2;

                // toggle
                enable = 1;
                usleep(300000);
                rainbow_line_printf("BOT: ON\n");
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                rainbow_line_printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // bot on/off
        if(enable == 1)
        {
            // always tracks sps
            static uint64_t st = 0;
            if(microtime() - st >= 1000000)
            {
                if(key_is_pressed(XK_H))
                    printf("\e[36mSPS: %u\e[0m\n", sps);
                sps = 0;
                st = microtime();
            }

            // input toggle
            if(key_is_pressed(XK_Control_R) && key_is_pressed(XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(300000);
                    printf("HOTKEYS: ON [%ix%i]\n", x, y);
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(300000);
                    rainbow_line_printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // crosshair toggle
                if(key_is_pressed(XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        rainbow_line_printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }

                // sample capture toggle
                if(key_is_pressed(XK_L))
                {
                    if(sample_capture == 0)
                    {
                        char* home = getenv("HOME");
                        sprintf(targets_dir, "%s/Desktop/targets", home);
                        mkdir(targets_dir, 0777);

                        sample_capture = 1;
                        usleep(300000);
                        rainbow_line_printf("SAMPLE CAPTURE: ON\n");
                        speakS("sc on");
                    }
                    else
                    {
                        sample_capture = 0;
                        usleep(300000);
                        rainbow_line_printf("SAMPLE CAPTURE: OFF\n");
                        speakS("sc off");
                    }
                }

                if(key_is_pressed(XK_G)) // print activation when pressed
                {
                    const float ret = processModel();
                    if(ret > 0.f)
                    {
                        if(ret >= ACTIVATION_SENITIVITY)
                        {
                            printf("\e[93mA: %f\e[0m\n", ret);
                            XSetForeground(d, gc, 65280);
                            XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                            XSetForeground(d, gc, 0);
                            XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                            XFlush(d);
                        }
                        else
                        {
                            const uint s = (uint)((1.f-ret)*255.f);
                            printf("\x1b[38;2;255;%u;%um A: %f\n", s, s, ret);
                            XSetForeground(d, gc, 16711680);
                            XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                            XSetForeground(d, gc, 0);
                            XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                            XFlush(d);
                        }
                    }
                }
            }

            const float activation = processModel();
            if(activation >= ACTIVATION_SENITIVITY)
            {
                tc++;

                // did we activate enough times in a row to be sure this is a target?
                if(tc > REPEAT_ACTIVATION)
                {
                    if(sample_capture == 1)
                    {
                        char name[32];
                        sprintf(name, "%s/%i.ppm", targets_dir, rand());
                        saveSample(twin, name);
                    }

                    xdo_mouse_down(xdo, CURRENTWINDOW, 1);
                    usleep(100000); // or cheating ban
                    xdo_mouse_up(xdo, CURRENTWINDOW, 1);

                    if(sample_capture == 1)
                        usleep(350000); //sleep(1);

                    // display ~1s recharge time
                    if(crosshair != 0)
                    {
                        crosshair = 2;
                        ct = time(0);
                    }
                }
            }
            else
            {
                tc = 0;
            }

            if(crosshair == 1)
            {
                if(sample_capture == 1)
                {
                    if(draw_sa > 0)
                    {
                        // draw sample outline
                        XSetForeground(d, gc, 16711680);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 16711680);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                        draw_sa -= 1;
                    }
                    else
                    {
                        XSetForeground(d, gc, 16777215);
                        XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                        XSetForeground(d, gc, 16776960);
                        XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                        XFlush(d);
                    }
                }
                else
                {
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XFlush(d);
                }
            }

            if(crosshair == 2)
            {
                if(time(0) > ct+1)
                    crosshair = 1;

                if(sample_capture == 1)
                {
                    XSetForeground(d, gc, 16777215);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }
                else
                {
                    XSetForeground(d, gc, 16711680);
                    XDrawRectangle(d, twin, gc, x-sw2-1, y-sh2-1, sw+2, sh+2);
                    XSetForeground(d, gc, 65280);
                    XDrawRectangle(d, twin, gc, x-sw2-2, y-sh2-2, sw+4, sh+4);
                    XFlush(d);
                }
            }
        }

        //
    }

    // done, never gets here in regular execution flow
    XCloseDisplay(d);
    return 0;
}


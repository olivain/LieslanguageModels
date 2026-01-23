#include <Arduino.h>
#include <GxEPD2_BW.h>
#include <gdey/GxEPD2_370_GDEY037T03.h> 
#define EPD_CS    15 
#define EPD_DC    4  
#define EPD_RST   2  
#define EPD_BUSY  5  


#define GxEPD2_DISPLAY_CLASS GxEPD2_BW
#define GxEPD2_DRIVER_CLASS GxEPD2_370_GDEY037T03

#define MAX_DISPLAY_BUFFER_SIZE 12480ul
#define MAX_HEIGHT(EPD) (EPD::HEIGHT <= MAX_DISPLAY_BUFFER_SIZE / (EPD::WIDTH / 8) ? EPD::HEIGHT : MAX_DISPLAY_BUFFER_SIZE / (EPD::WIDTH / 8))

GxEPD2_DISPLAY_CLASS<GxEPD2_DRIVER_CLASS, MAX_HEIGHT(GxEPD2_DRIVER_CLASS)> display(GxEPD2_DRIVER_CLASS(EPD_CS, EPD_DC, EPD_RST, EPD_BUSY));

unsigned char imageBuffer[MAX_DISPLAY_BUFFER_SIZE];

void displayString(String ipText)
{
    display.setFullWindow();
    display.firstPage();
    do
    {
        yield(); // <--- CRITICAL for ESP8266
        display.fillScreen(GxEPD_WHITE);
        display.setCursor(0, 20);
        display.setTextColor(GxEPD_BLACK);
        display.setFont(NULL);  // Use default font
        display.setTextSize(1); // Adjust size if needed
        display.print(ipText);
    } while (display.nextPage());

    display.powerOff();
}

void setup()
{
    delay(500);

    Serial.begin(230400); // 9600 for serial monitoring on platformio (cf platformio.ini)

    delay(1500);

    // Serial.println("hello world");
    display.init();
    display.setRotation(2); // adjust as needed

    delay(100);

    displayString("Lies Language Models - O. Porry - 2026");

    delay(3000);

    displayString("Les menteurs ne gagnent qu'une chose. C'est de ne pas être crus, même lorsqu'ils disent la vérité.");

    delay(500);

    Serial.println("ESP8266 IS READY TO RECEIVE IMG !");
}
void loop()
{
    static enum {
        WAIT_HEADER,
        WAIT_IMAGE
    } state = WAIT_HEADER;

    static uint32_t expectedLength = 0;
    static uint32_t receivedBytes = 0;

    // ---------- STATE: WAIT FOR HEADER ----------
    if (state == WAIT_HEADER)
    {
        if (Serial.available() >= 4)
        {
            expectedLength = 0;
            for (int i = 0; i < 4; i++)
            {
                expectedLength = (expectedLength << 8) | Serial.read();
            }

            Serial.print("Incoming image size: ");
            Serial.println(expectedLength);

            if (expectedLength > MAX_DISPLAY_BUFFER_SIZE)
            {
                Serial.println("ERROR: Image too large, discarding.");
                expectedLength = 0;
                receivedBytes = 0;
                return;
            }

            receivedBytes = 0;
            state = WAIT_IMAGE;
        }
    }

    // ---------- STATE: RECEIVE IMAGE DATA ----------
    if (state == WAIT_IMAGE)
    {
        while (Serial.available() && receivedBytes < expectedLength)
        {
            imageBuffer[receivedBytes++] = Serial.read();
            yield(); // ESP8266 watchdog protection
        }

        if (receivedBytes == expectedLength)
        {
            Serial.println("Image received, rendering...");

            display.setFullWindow();
            display.firstPage();
            do
            {
                yield(); // CRITICAL on ESP8266

                display.fillScreen(GxEPD_WHITE);

                display.drawInvertedBitmap(
                        0, 0,
                        imageBuffer,
                        display.width(),
                        display.height(),
                        GxEPD_BLACK
                    );


            } while (display.nextPage());

            display.powerOff();

            Serial.println("Display update complete.");

            // Reset state machine
            state = WAIT_HEADER;
            expectedLength = 0;
            receivedBytes = 0;
        }
    }
}

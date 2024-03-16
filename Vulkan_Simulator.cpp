// Vulkan Simulator.cpp : Defines the entry point for the application.
//

#include "Vulkan_Simulator.h"

Q_LOGGING_CATEGORY(lcVk, "qt.vulkan")



VulkanWindow::VulkanWindow()
{
    qDebug("VulkanWindow creator");

    setSurfaceType(QSurface::VulkanSurface);
}

VulkanWindow::~VulkanWindow()
{
    qDebug("VulkanWindow deletor");

    delete mTimer;
    release();
}

void VulkanWindow::release()
{
    qDebug("release");

    if (!mDevice) return;

    vkDeviceWaitIdle(mDevice);

    if (mCommandPool)
    {
        vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
        mCommandPool = VK_NULL_HANDLE;
    }

    if (mDevice)
    {
        vkDestroyDevice(mDevice, nullptr);

        // Play nice and notify QVulkanInstance that the QVulkanDeviceFunctions
        // for mDevice needs to be invalidated.
     ///   vulkanInstance()->resetDeviceFunctions(mDevice);  // TODO: destroy everything before using this!
        mDevice = VK_NULL_HANDLE;
    }

    mSurface = VK_NULL_HANDLE;
}

/// <summary>
///Returns a QMatrix4x4 that can be used to correct for coordinate
///system differences between OpenGL and Vulkan.
///
///By pre - multiplying the projection matrix with this matrix, applications can
///continue to assume that Y is pointing upwards, and can set minDepthand
///maxDepth in the viewport to 0 and 1, respectively, without having to do any
///further corrections to the vertex Z positions.Geometry from OpenGL
///applications can then be used as - is, assuming a rasterization state matching
///the OpenGL cullingand front face settings.
/// </summary>
/// <returns></returns>
QMatrix4x4 VulkanWindow::clipCorrectionMatrix()
{

    if (mClipCorrect.isIdentity()) {

        // The QMatrix creator takes row-major.
        mClipCorrect = QMatrix4x4(1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, -1.0f, 0.5f,
            0.0f, 0.0f, 0.0f, 1.0f);
    }
    return mClipCorrect;
}

void VulkanWindow::setupVulkanInstance(QVulkanInstance& instance) {
    mQInstance = &instance;
    mInstance = instance.vkInstance();

    setVulkanInstance(&instance);

    QByteArrayList extensions = { "VK_EXT_debug_utils" };  // This doesn't work. Function vkSetDebugUtilsObjectNameEXT remains a null pointer.
    instance.setExtensions(extensions);

    // Get window and Vulkan instance functions.
    mVulkanPointers.pVulkanWindow = this;
    mVulkanPointers.pVulkanFunctions = vulkanInstance()->functions();
    mVulkanPointers.pInstance = vulkanInstance();

    // Let user select a glTF file to be shown.
    QString filename = QFileDialog::getOpenFileName(
        nullptr,
        QObject::tr("Open glTF file"),
#ifdef __linux__
        "/home",
#elif _WIN32
        "C:\\Users\\Public",
#else
        QDir::currentPath(),
#endif 
        QObject::tr("glTF files (*.gltf)"));
    mVulkanPointers.path = filename.toStdString();

    // Setup Vulkan device, physical device, queues and command pool.
    init();

    // Load glTF file and setup other needed Vulkan stuff.
    initResources();

    // Start VulkanWindow refresh timer. There seems not to be a possibility to use std::unique_ptr
    // with QTimer, so let's use the good old wintage methods new and delete.
    mTimer = new QTimer();
    connect(mTimer, &QTimer::timeout, this, &VulkanWindow::refresh);
    mTimer->start(50);
}

/// <summary>
/// This helper function initializes necessary components inherently coupled with VkInstace:
/// device, physical device, queues and command pool.
/// </summary>
void VulkanWindow::init()
{
    qDebug("init");

    QVulkanInstance* inst = vulkanInstance();
    mSurface = QVulkanInstance::surfaceForWindow(this);
    if (!mSurface) qFatal("Failed to get surface for window.");

    // Enumerate Vulkan physical devices.
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
    if (deviceCount == 0) qFatal("No physical Vulkan devices.");
    QVector<VkPhysicalDevice> physDevices(deviceCount);
    VkResult err = vkEnumeratePhysicalDevices(mInstance, &deviceCount, physDevices.data());
    if (err != VK_SUCCESS && err != VK_INCOMPLETE)
        qFatal("Failed to enumerate Vulkan physical devices: %d", err);

    // Select suitable physical device.
    int integrated = -1;
    int discrete = -1;
    for (int i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties physDeviceProps;
        vkGetPhysicalDeviceProperties(physDevices[i], &physDeviceProps);

        if (physDeviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && discrete == -1) {
            discrete = i;
            qDebug("Discrete device name: %s Driver version: %d.%d.%d",
                physDeviceProps.deviceName,
                VK_API_VERSION_MAJOR(physDeviceProps.driverVersion),
                VK_API_VERSION_MINOR(physDeviceProps.driverVersion),
                VK_API_VERSION_PATCH(physDeviceProps.driverVersion));
        }

        if (physDeviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && integrated == -1) {
            integrated = i;
            qDebug("Integrated device name: %s Driver version: %d.%d.%d",
                physDeviceProps.deviceName,
                VK_API_VERSION_MAJOR(physDeviceProps.driverVersion),
                VK_API_VERSION_MINOR(physDeviceProps.driverVersion),
                VK_API_VERSION_PATCH(physDeviceProps.driverVersion));
        }
    }
    mPhysDevice = physDevices[0];
    if (integrated != -1) mPhysDevice = physDevices[integrated];
    if (discrete != -1) mPhysDevice = physDevices[discrete];

    // Setup function pointers regarding to the physical device.
    initPhysDeviceFunctions();

    // Graphics drivers executes commands asychronously along queues, which themselves are grouped
    // into queue families. We need to select graphics queue and presentation queue.
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysDevice, &queueCount, nullptr);
    QVector<VkQueueFamilyProperties> queueFamilyProps(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysDevice, &queueCount,
        queueFamilyProps.data());
    int graphicsQueueFamilyIndex = -1;
    int presentQueueFamilyIndex = -1;

    // First look for a queue family that supports both.
    for (int i = 0; i < queueFamilyProps.count(); ++i)
    {
        if (graphicsQueueFamilyIndex == -1 &&
            (queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            inst->supportsPresent(mPhysDevice, i, this))
            graphicsQueueFamilyIndex = i;
    }

    if (graphicsQueueFamilyIndex != -1) {
        presentQueueFamilyIndex = graphicsQueueFamilyIndex;
    }
    else {

        // Separate queues then.
        qDebug("No queue with graphics+present; Trying separate queues.");
        for (int i = 0; i < queueFamilyProps.count(); ++i) {
            if (graphicsQueueFamilyIndex == -1 &&
                (queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                graphicsQueueFamilyIndex = i;
            if (presentQueueFamilyIndex == -1 &&
                inst->supportsPresent(mPhysDevice, i, this))
                presentQueueFamilyIndex = i;
        }
    }
    if (graphicsQueueFamilyIndex == -1)
        qFatal("No graphics queue family found");
    if (presentQueueFamilyIndex == -1)
        qFatal("No present queue family found");

    // Create logical device.
    VkDeviceQueueCreateInfo queueInfo[2];
    const float prio[] = { 0 };
    memset(queueInfo, 0, sizeof(queueInfo));
    queueInfo[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo[0].queueFamilyIndex = graphicsQueueFamilyIndex;
    queueInfo[0].queueCount = 1;
    queueInfo[0].pQueuePriorities = prio;

    if (graphicsQueueFamilyIndex != presentQueueFamilyIndex) {
        queueInfo[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo[1].queueFamilyIndex = presentQueueFamilyIndex;
        queueInfo[1].queueCount = 1;
        queueInfo[1].pQueuePriorities = prio;
    }

    QVector<const char*> deviceExtensions;
    deviceExtensions.append("VK_KHR_swapchain");

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = graphicsQueueFamilyIndex == presentQueueFamilyIndex ? 1 : 2;
    devInfo.pQueueCreateInfos = queueInfo;
    devInfo.enabledLayerCount = ValidationLayers.size();
    devInfo.ppEnabledLayerNames = ValidationLayers.data();
    devInfo.enabledExtensionCount = deviceExtensions.count();
    devInfo.ppEnabledExtensionNames = deviceExtensions.constData();

    err = vkCreateDevice(mPhysDevice, &devInfo, nullptr, &mDevice);
    if (err != VK_SUCCESS) qFatal("Failed to create logical device: %d", err);

    // Setup function pointers regarding to the logical device.
    initDeviceFunctions();

    // Create command pool.
    vkGetDeviceQueue(mDevice, graphicsQueueFamilyIndex, 0, &mGraphicsQueue);
    if (graphicsQueueFamilyIndex == presentQueueFamilyIndex) {
        mPresentQueue = mGraphicsQueue;
    }
    else {
        vkGetDeviceQueue(mDevice, presentQueueFamilyIndex, 0, &mPresentQueue);
    }

}



void VulkanWindow::initDeviceFunctions() {
    vkCreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(
        vkGetDeviceProcAddr(mDevice, "vkCreateSwapchainKHR"));

    vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
        vkGetDeviceProcAddr(mDevice, "vkDestroySwapchainKHR"));

    vkGetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(
        vkGetDeviceProcAddr(mDevice, "vkGetSwapchainImagesKHR"));

    vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
        vkGetDeviceProcAddr(mDevice, "vkAcquireNextImageKHR"));

    vkQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(
        vkGetDeviceProcAddr(mDevice, "vkQueuePresentKHR"));
}

void VulkanWindow::initPhysDeviceFunctions() {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            vkGetInstanceProcAddr(mInstance,
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));

    vkGetPhysicalDeviceSurfaceFormatsKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));

    vkGetPhysicalDeviceSurfacePresentModesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
            vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceSurfacePresentModesKHR"));

    vkGetPhysicalDeviceQueueFamilyProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceQueueFamilyProperties"));

    vkGetPhysicalDeviceSurfaceSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceSurfaceSupportKHR"));

    vkGetPhysicalDeviceFeatures =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
            vkGetInstanceProcAddr(mInstance, "vkGetPhysicalDeviceFeatures"));

    vkEnumerateDeviceExtensionProperties =
        reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            vkGetInstanceProcAddr(mInstance, "vkEnumerateDeviceExtensionProperties"));
}

void VulkanWindow::refresh() {
    if (!mStart && mInitialized)
        mRenderer->render();
}

void VulkanWindow::resizeEvent(QResizeEvent*) {
    // TODO: Implement this.
}

void VulkanWindow::exposeEvent(QExposeEvent*) {
    if (isExposed() && !mInitialized)
    {
        mInitialized = true;
        initSwapChainResources();
        mRenderer->render();
    }

    // Release everything when unexposed - the meaning of which is platform
    // specific. Can be essential on mobile, to release resources while in
    // background.
    if (!isExposed() && mInitialized) {
        mInitialized = false;
        releaseSwapChainResources();
        releaseResources();
    }
}

bool VulkanWindow::event(QEvent* e) {

    if (mRenderer != nullptr && mStart == true) {
        mStart = false;
        initSwapChainResources();
        mRenderer->render();
    }

    switch (e->type())
    {
    case QEvent::UpdateRequest:
        mRenderer->render();
        break;

        // The swapchain must be destroyed before the surface as per spec.
        // This is not ideal for us because the surface is managed by the
        // QPlatformWindow which may be gone already when the unexpose comes, making
        // the validation layer scream. The solution is to listen to the
        // PlatformSurface events.
    case QEvent::PlatformSurface: {
        auto* ev = static_cast<QPlatformSurfaceEvent*>(e);
        if (ev->surfaceEventType() == QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed) {
            releaseSwapChainResources();
            releaseResources();
        }
        break;
    }

    default:
        break;
    }

    return QWindow::event(e);
}

/// <summary>
/// This helper function loads glTF file and creates Renderer class instance, which sets the most
/// of Vulkan stuff. Before calling this function call init() first! Also pVulkanWindow and 
/// pInstance as well as glTF file path need to be set before. 
/// </summary>
void VulkanWindow::initResources()
{
    qDebug("initResources");

    // Get device and Vulkan device functions.
    mVulkanPointers.device = mVulkanPointers.pVulkanWindow->device();
    mVulkanPointers.physicalDevice = mVulkanPointers.pVulkanWindow->physicalDevice();
    mVulkanPointers.pDeviceFunctions = mVulkanPointers.pVulkanWindow->vulkanInstance()->
        deviceFunctions(mVulkanPointers.pVulkanWindow->device());
    mVulkanPointers.surface = mVulkanPointers.pInstance->surfaceForWindow(
        mVulkanPointers.pVulkanWindow);

    // Load 3D model into memory.
    if (mFileReader == nullptr) {
        mFileReader = std::make_unique<FileReader>();
    }
    if (!mFileReader->loadFile(mVulkanPointers.path))
        qFatal("Couldn't load a file!");

    // Create renderer and init all the permanent parts of the renderer.
    mRenderer = std::make_unique<Renderer>(Renderer(mVulkanPointers));
    mRenderer->createVertexBuffer(*mFileReader->getModel());
    mRenderer->setViewMatrix();
    mRenderer->setModelMatrix();
    mRenderer->createUniformBuffers();
    mRenderer->createGraphicsPipeline();
    mRenderer->createSyncObjects();
}

/// <summary>
/// This helper function creates the parts of Vulkan which may be destroyed and recreated during
/// the application lifetime: swapchain, imageviews and renderpass.
/// </summary>
void VulkanWindow::initSwapChainResources()
{
    qDebug("initSwapChainResources");

    // Every window recreate need to set projection matrix in a case user resizes window.
    // Use this...
    QMatrix4x4 proj1 = clipCorrectionMatrix();
    const QSize size = swapChainImageSize();
    proj1.perspective(45.0f, size.width() / (float)size.height(), 0.1f, 1000.0f);



    // Perspective projection parameters.
    // ...or this.
    float fov = 45.0f;
    float aspectRatio = size.width() / (float)size.height();
    float nearZ = 0.1f;
    float farZ = 1000.0f;
    float top = nearZ * std::tan(M_PI / 180 * fov / 2.0);
    float bottom = -top;
    float right = top * aspectRatio;
    float left = -right;
    QMatrix4x4 proj2 = QMatrix4x4(2 * nearZ / (right - left), 0, (right + left) / (right - left), 0,
        0, 2 * nearZ / (top - bottom), (top + bottom) / (top - bottom), 0,
        0, 0, -(farZ + nearZ) / (farZ - nearZ), -(2 * farZ * nearZ) / (farZ - nearZ),
        0, 0, -1, 0);

    proj2 = proj2 * clipCorrectionMatrix();

    mRenderer->setProjectionMatrix(proj2.data());

    // Then create swap chain.
    mRenderer->createSwapChain(nullptr, nullptr, mRenderer->getSwapChain());
}

void VulkanWindow::releaseSwapChainResources()
{
    qDebug("releaseSwapChainResources");

    // It is important to finish the pending frame right here since this is the
    // last opportunity to act with all resources intact.
    vkDeviceWaitIdle(mDevice);

    mRenderer->deleteSwapChain();
}


void VulkanWindow::releaseResources()
{
    qDebug("releaseResources");

    // Before releasing resources it is important there aren't any pending work going.
    mVulkanPointers.pDeviceFunctions->vkDeviceWaitIdle(mVulkanPointers.device);

    mRenderer->deleteVertexBuffer();
    mRenderer->deleteUniformBuffers();
    mRenderer->deleteGraphicsPipeline();
    mRenderer->deleteSwapChain();
    mRenderer->deleteSyncObjects();
}

VkInstance VulkanWindow::createInstance() {

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Simulator";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = nullptr;//"No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    QVector<const char*> reqExtensions = getRequiredExtensions();
    instInfo.enabledExtensionCount = reqExtensions.size();
    instInfo.ppEnabledExtensionNames = reqExtensions.data();
    if (enableValidationLayers)
    {
        instInfo.enabledLayerCount = ValidationLayers.size();
        instInfo.ppEnabledLayerNames = ValidationLayers.data();
    }
    else {
        instInfo.enabledLayerCount = 0;
    }

    VkInstance instance;
    if (vkCreateInstance(&instInfo, nullptr, &instance) == VK_SUCCESS)
        return instance;

    qFatal("Failed to create Vulkan instance!");
    return VK_NULL_HANDLE;
}

QVector<const char*> VulkanWindow::getRequiredExtensions() {
    QVector<const char*> extensions;
    extensions.append(VK_KHR_SURFACE_EXTENSION_NAME);// "VK_KHR_surface"

    // Vulkan surface types:
    //VK_KHR_win32_surface
    //VK_KHR_wayland_surface
    //VK_KHR_xcb_surface
    //VK_KHR_xlib_surface
    //VK_KHR_android_surface
    //VK_MVK_macos_surface
    //VK_MVK_ios_surface

#ifdef _WIN32
    extensions.append("VK_KHR_win32_surface");
#elif __APPLE__
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR

     // iOS Simulator
    mtl_line();
#elif TARGET_OS_IPHONE

    // iOS device
    extensions.append("VK_MVK_ios_surface");
#elif TARGET_OS_MAC

    // Other kinds of Mac OS
    extensions.append("VK_MVK_macos_surface");
#else
    qWarning("Unknown Apple platform");
#endif
#else
    extensions.append(pickLinuxSurfaceExtension());
#endif

    if (enableValidationLayers)
        extensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

    return extensions;
}

LinuxDisplayType VulkanWindow::getLinuxDisplayType() {
    // Inspired from:
    // stackoverflow.com/questions/45536141/how-i-can-find-out-if-a-linux-system-uses-wayland-or-x11

    auto env = QProcessEnvironment::systemEnvironment();

    QString value = env.value(QLatin1String("WAYLAND_DISPLAY"));
    if (!value.isEmpty())
        return LinuxDisplayType::Wayland;

    value = env.value(QLatin1String("DISPLAY"));
    if (!value.isEmpty())
        return LinuxDisplayType::X11;

    qWarning("Unknown Linux display type");
    return LinuxDisplayType::None;
}

const char* VulkanWindow::pickLinuxSurfaceExtension() {

#ifdef __ANDROID__
    return "VK_KHR_android_surface";
#endif

    const auto display = getLinuxDisplayType();

    if (display == LinuxDisplayType::Wayland)
        return "VK_KHR_wayland_surface";

    if (display == LinuxDisplayType::X11)
        return "VK_KHR_xcb_surface";

    qWarning("No Linux surface extension");
    return nullptr;
}

/// <summary>
/// For Vulkan debugging we need some layers installed. Global vector ValidationLayers
/// tells what layers we want to install. CheckValidationLayerSupport checks are those
/// layers available.
/// </summary>
/// <param name="inst">Optional QVulkanInstance parameter.</param>
/// <returns></returns>
bool CheckValidationLayerSupport(QVulkanInstance* inst = nullptr) {

    if (inst != nullptr) {

        // First enumerate available layers.
        QVulkanInfoVector<QVulkanLayer> layers = inst->supportedLayers();

        // Check if there are the layers we want among availabale layers.
        for (const char* layerName : ValidationLayers) {
            bool layerFound = false;
            for (const auto& layer : layers) {
                if (strcmp(layerName, layer.name) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }
    }
    else {

        // First enumerate available layers.
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // Check if there are the layers we want among available layers.
        for (const char* layerName : ValidationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[])
{

    // Let's get additional Qt diagnosis data to the Linux Console Window.
    // This affects only for a Linux build.
    qputenv("QT_DEBUG_PLUGINS", QByteArray("1"));

    // QApplication is the child class of QGuiApplication which has base class QCoreApplication. 
    // We need QApplication because we want to use both QWidgets and QVulkanInstance.
    QApplication app(argc, argv);
    QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));

    // Create a window to render into.
    VulkanWindow w;
    w.resize(1024, 768);
    w.show();

    // If we are running on debug mode, create Vulkan validation layers. 
    QVulkanInstance inst;
    if (enableValidationLayers && !CheckValidationLayerSupport(&inst)) {
        qWarning("Validation layers requested, but not available!");
    }
    else {
        qInfo("Validation layers requested.");
        QList<QByteArray> temp;
        temp.reserve(ValidationLayers.size());
        for (int i = 0; i < ValidationLayers.size(); i++) {
            temp.append(ValidationLayers[i]);
        }
        inst.setLayers(temp);
    }

    // Install needed Vulkan extensions.
    auto extra = w.getRequiredExtensions();
    QByteArrayList list;
    for (const auto& next : extra) list.append(next);
    inst.setExtensions(list);

    // Assing our own self created Vulkan instance for QVulkanInstance so that it doesn't
    // make its' default instance.
    VkInstance instance = w.createInstance();
    if (instance == VK_NULL_HANDLE)
    {
        qFatal("Failed to create VkInstance!");
    }
    inst.setVkInstance(instance);

    // Every Vulkan-based QVulkanWindow must be associated with a QVulkanInstance. Our custom
    // VulkanWindow does not make a difference with default QVulkanWindow, so let's create one.
    if (!inst.create())
        qFatal("Failed to create QVulkanInstance: %d", inst.errorCode());

    // Connect QVulkanInstance with VulkanWindow. This also loads other Vulkan stuff and let
    // user choose gltf file to show on window.
    w.setupVulkanInstance(inst);

    // Finally start application.
    return app.exec();
}

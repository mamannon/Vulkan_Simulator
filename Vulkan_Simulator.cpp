
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
    this->release();
}

void VulkanWindow::release()
{
    qDebug("release");

    if (!mVulkanPointers.device) return;

    vkDeviceWaitIdle(mVulkanPointers.device);

    // Play nice and notify QVulkanInstance that the QVulkanDeviceFunctions
    // for mDevice needs to be invalidated.
    ///vulkanInstance()->resetDeviceFunctions(mVulkanPointers.device);  // DOESN'T WORK, LET BE.

    // Delete Vulkan logical device.
    if (mVulkanPointers.device)
    {
        vkDestroyDevice(mVulkanPointers.device, nullptr);
        if (enableValidationLayers) {
            this->releaseDebugMesseger();
        }
        mVulkanPointers.device = VK_NULL_HANDLE;
    }

    // Delete Vulkan instance.
    vkDestroyInstance(mVulkanPointers.instance, nullptr);
    mVulkanPointers.instance = VK_NULL_HANDLE;

    mVulkanPointers.surface = VK_NULL_HANDLE;
}

/// <summary>
/// SetupVulkanInstance function collects together other Vulkan initialization functions. It needs a valid
/// initalized Vulkan instance as a parameter.
/// </summary>
/// <param name="instance">Qt libaray wrapper for Vulkan instance.</param>
void VulkanWindow::setupVulkanInstance(QVulkanInstance& instance) {
    this->setVulkanInstance(&instance);

    // Get window, surface and Vulkan instance function pointers.
    mVulkanPointers.vulkanWindow = this->shared_from_this();
    mVulkanPointers.pVulkanFunctions = this->vulkanInstance()->functions();
    mVulkanPointers.pInstance = &instance;
    mVulkanPointers.instance = instance.vkInstance();
    mVulkanPointers.surface = this->vulkanInstance()->surfaceForWindow(this);
    mFileReader = std::make_shared<FileReader>();
    mVulkanPointers.fileReader = std::weak_ptr<FileReader>(mFileReader);

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
    this->init();

    // Load glTF file and setup other needed Vulkan stuff.
    this->initResources();

    // Start VulkanWindow refresh timer. There seems not to be a possibility to use std::unique_ptr
    // with QTimer, so let's use the good old wintage methods new and delete.
    mTimer = new QTimer();
    connect(mTimer, &QTimer::timeout, this, &VulkanWindow::refresh);
    mTimer->start(50);
}

/// <summary>
/// Init function initializes necessary components inherently coupled with VkInstace:
/// device, physical device, queues and Renderer class instance.
/// </summary>
void VulkanWindow::init()
{
    qInfo("init");

    QVulkanInstance* inst = this->vulkanInstance();
    mVulkanPointers.surface = QVulkanInstance::surfaceForWindow(this);
    if (!mVulkanPointers.surface) qFatal("Failed to get surface for window.");

    // Setup function pointers regarding to the physical device.
    this->initPhysDeviceFunctions();

    if (enableValidationLayers) {

        //If we want to use custom debug messenger.
        this->setupDebugMessenger();
    }

    // Enumerate Vulkan physical devices.
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mVulkanPointers.instance, &deviceCount, nullptr);
    if (deviceCount == 0) qFatal("No physical Vulkan devices.");
    std::vector<VkPhysicalDevice> physDevices(deviceCount);
    VkResult err = vkEnumeratePhysicalDevices(mVulkanPointers.instance, &deviceCount, physDevices.data());
    if (err != VK_SUCCESS && err != VK_INCOMPLETE)
        qFatal("Failed to enumerate Vulkan physical devices: %d", err);

    // Select suitable physical device.
    int integrated = -1;
    int discrete = -1;
    QueueFamilyIndices integratedQFI{};
    QueueFamilyIndices discreteQFI{};
    SwapChainSupportDetails integratedSCSD{};
    SwapChainSupportDetails discreteSCSD{};
    for (int i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties physDeviceProps{};
        vkGetPhysicalDeviceProperties(physDevices[i], &physDeviceProps);

        // Do we have a discrete GPU?
        if (physDeviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && discrete == -1) {
            if (this->isDeviceSuitable(physDevices[i], mVulkanPointers.surface, discreteQFI, discreteSCSD)) {
                discrete = i;
                qDebug("Discrete device name: %s Driver version: %d.%d.%d",
                    physDeviceProps.deviceName,
                    VK_API_VERSION_MAJOR(physDeviceProps.driverVersion),
                    VK_API_VERSION_MINOR(physDeviceProps.driverVersion),
                    VK_API_VERSION_PATCH(physDeviceProps.driverVersion));
            }
        }

        // Do we have an integrated GPU?
        if (physDeviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && integrated == -1) {
            if (this->isDeviceSuitable(physDevices[i], mVulkanPointers.surface, integratedQFI, integratedSCSD)) {
                integrated = i;
                qDebug("Integrated device name: %s Driver version: %d.%d.%d",
                    physDeviceProps.deviceName,
                    VK_API_VERSION_MAJOR(physDeviceProps.driverVersion),
                    VK_API_VERSION_MINOR(physDeviceProps.driverVersion),
                    VK_API_VERSION_PATCH(physDeviceProps.driverVersion));
            }
        }

    }
    if (integrated != -1) {
        mVulkanPointers.physicalDevice = physDevices[integrated];
        mVulkanPointers.queueFamilyIndices = integratedQFI;
        mVulkanPointers.swapChainSupportDetails = integratedSCSD;
    }

    if (discrete != -1) {
        mVulkanPointers.physicalDevice = physDevices[discrete];
        mVulkanPointers.queueFamilyIndices = discreteQFI;
        mVulkanPointers.swapChainSupportDetails = discreteSCSD;
    }

    if (mVulkanPointers.physicalDevice == 0) {
        qFatal("Did not found any suitable physical device (graphics card)!");
    }

    // Create logical device.
    if (!mVulkanPointers.queueFamilyIndices.graphicsFamily.has_value() ||
        !mVulkanPointers.queueFamilyIndices.presentFamily.has_value()) {
        qFatal("Graphics queue family and/or presentation queue family missing!");
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> queueFamilies = { mVulkanPointers.queueFamilyIndices.graphicsFamily.value(),
        mVulkanPointers.queueFamilyIndices.presentFamily.value() };
    float prio = 1.0f;
    for (uint32_t queueFamily : queueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &prio;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = (uint32_t)
        mVulkanPointers.queueFamilyIndices.graphicsFamily.value() ==
        mVulkanPointers.queueFamilyIndices.presentFamily.value() ? 1 : 2;
    devInfo.pQueueCreateInfos = queueCreateInfos.data();
    if (enableValidationLayers) {
        devInfo.enabledLayerCount = validationLayers.size();
        devInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else {
        devInfo.enabledLayerCount = 0;
    }
    devInfo.enabledExtensionCount = deviceExtensions.size();
    devInfo.ppEnabledExtensionNames = deviceExtensions.data();

    // We don't need optional features in this application, so we give device features struct where all 
    // features are set to default IE. not using any features.
    VkPhysicalDeviceFeatures deviceFeatures{};
    devInfo.pEnabledFeatures = &deviceFeatures;

    err = vkCreateDevice(mVulkanPointers.physicalDevice, &devInfo, nullptr, &mVulkanPointers.device);
    if (err != VK_SUCCESS) qFatal("Failed to create logical device: %d", err);

    vkGetDeviceQueue(mVulkanPointers.device, mVulkanPointers.queueFamilyIndices.graphicsFamily.value(),
        0, &mVulkanPointers.graphicsQueue);
    vkGetDeviceQueue(mVulkanPointers.device, mVulkanPointers.queueFamilyIndices.presentFamily.value(),
        0, &mVulkanPointers.presentQueue);

    // Get Qt Vulkan Logical Device pointers.
    mVulkanPointers.pDeviceFunctions = this->vulkanInstance()->
        deviceFunctions(mVulkanPointers.device);

    // Setup function pointers regarding to the logical device.
    this->initDeviceFunctions();

    // Create Renderer class instance.
    if (!mVulkanPointers.swapChainSupportDetails.capabilities.has_value() ||
        mVulkanPointers.swapChainSupportDetails.formats.size() == 0 ||
        mVulkanPointers.swapChainSupportDetails.presentModes.size() == 0) {
        qFatal("SwapChainSupportDetails data missing!");
    }
    mRenderer = std::make_unique<Renderer>(Renderer(mVulkanPointers));

    // Init the necessary Vulkan stuff.
    mRenderer->setViewMatrix();
    mRenderer->setModelMatrix();
    mRenderer->createSyncObjects();
    mRenderer->createCommandPool();
    this->initSwapChainResources();
    mRenderer->createGraphicsPipeline();
    mRenderer->createUniformBuffers();
}

/// <summary>
/// InitResources loads glTF file and creates vertex buffers. Before calling 
/// this function call init() first! 
/// </summary>
void VulkanWindow::initResources()
{
    qInfo("initResources");

    // Load 3D model into memory.
    if (!mFileReader->loadFile(mVulkanPointers.path))
        qFatal("Couldn't load a file!");

    // Create vertex data from opened file.
    mRenderer->createVertexBuffer(*mFileReader->getModel());
}

/// <summary>
/// SetupDegugMesseger function creates a custom debug messenger, which sends Vulkan degug messages to 
/// qInfo output starting with a text "Vulkan validation layer".
/// </summary>
void VulkanWindow::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    if (mVulkanPointers.vkCreateDebugUtilsMessengerEXT(
        mVulkanPointers.instance, &debugCreateInfo, nullptr, &mDebugMessenger) != VK_SUCCESS) {
        qWarning("Failed to set up debug messenger!");
    }
}

void VulkanWindow::releaseDebugMesseger() {
    mVulkanPointers.vkDestroyDebugUtilsMessengerEXT(mVulkanPointers.instance, mDebugMessenger, nullptr);
}

/// <summary>
/// IsDeviceSuitable function checks if the given device is suitable for us. 
/// All Vulkan graphics cards are not suitable for us. 
/// </summary>
/// <param name="device">VkPhysicalDevice handle.</param>
/// <param name="surface">Window we render into.</param>
/// <returns>Did it succeed?</returns>
bool VulkanWindow::isDeviceSuitable(const VkPhysicalDevice& device,
    const VkSurfaceKHR& surface, QueueFamilyIndices& qfi, SwapChainSupportDetails& scsd)
{

    // Need to find queue families supporting drawing commands and the ones supporting presentation.
    // Hopefully they are the same, but it is not necessary.
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    mVulkanPointers.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    mVulkanPointers.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        mVulkanPointers.vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }
        if (indices.graphicsFamily.has_value() && indices.presentFamily.has_value()) {
            break;
        }
        i++;
    }

    // Check if the device supports all the extensions we need.
    bool extensionsSupported = false;
    uint32_t extensionCount;
    mVulkanPointers.vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    mVulkanPointers.vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    std::set<std::string> requiredExtensions;
    for (const char* next : deviceExtensions) {
        if (next == "__linux__") {
            requiredExtensions.insert(getLinuxDisplayType());
        }
        else {
            requiredExtensions.insert(next);
        }
    }
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    extensionsSupported = requiredExtensions.empty();

    // Check if the device supports the swap chain we need: one supported image format 
    // and one supported presentation mode given the window surface we have.
    bool swapChainAdequate = false;
    SwapChainSupportDetails details{};
    if (extensionsSupported) {
        VkSurfaceCapabilitiesKHR temp;
        mVulkanPointers.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &temp);
        details.capabilities = temp;
        uint32_t formatCount;
        mVulkanPointers.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            mVulkanPointers.vkGetPhysicalDeviceSurfaceFormatsKHR(
                device, surface, &formatCount, details.formats.data());
        }
        uint32_t presentModeCount;
        mVulkanPointers.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            mVulkanPointers.vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface, &presentModeCount, details.presentModes.data());
        }
        swapChainAdequate = !details.formats.empty() && !details.presentModes.empty();
    }

    // Finally return are all the conditions suitable.
    bool temp = indices.graphicsFamily.has_value() && indices.presentFamily.has_value() &&
        extensionsSupported && swapChainAdequate;
    if (temp) {

        // Save queue family indices and swapchainsupportdetails for further use.
        qfi = indices;
        scsd = details;
    }
    return temp;
}

/// <summary>
/// InitDeviceFunctions function initializes Vulkan Logical Device function pointers.
/// </summary>
void VulkanWindow::initDeviceFunctions() {
    qInfo("initDeviceFunctions");

    mVulkanPointers.vkCreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkCreateSwapchainKHR"));

    mVulkanPointers.vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkDestroySwapchainKHR"));

    mVulkanPointers.vkGetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkGetSwapchainImagesKHR"));

    mVulkanPointers.vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkAcquireNextImageKHR"));

    mVulkanPointers.vkQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkQueuePresentKHR"));

    mVulkanPointers.vkDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(
        vkGetDeviceProcAddr(mVulkanPointers.device, "vkDeviceWaitIdle"));
}

/// <summary>
/// InitPhysDeviceFunctions function initializes Vulkan Physical Device (graphics card) function pointers.
/// </summary>
void VulkanWindow::initPhysDeviceFunctions() {
    qInfo("initPhysDeviceFunctions");

    mVulkanPointers.vkGetPhysicalDeviceSurfaceCapabilitiesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            vkGetInstanceProcAddr(mVulkanPointers.instance,
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));

    mVulkanPointers.vkGetPhysicalDeviceSurfaceFormatsKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));

    mVulkanPointers.vkGetPhysicalDeviceSurfacePresentModesKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"));

    mVulkanPointers.vkGetPhysicalDeviceQueueFamilyProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkGetPhysicalDeviceQueueFamilyProperties"));

    mVulkanPointers.vkGetPhysicalDeviceSurfaceSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));

    mVulkanPointers.vkGetPhysicalDeviceFeatures =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkGetPhysicalDeviceFeatures"));

    mVulkanPointers.vkEnumerateDeviceExtensionProperties =
        reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkEnumerateDeviceExtensionProperties"));

    mVulkanPointers.vkCreateDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkCreateDebugUtilsMessengerEXT"));

    mVulkanPointers.vkDestroyDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(mVulkanPointers.instance, "vkDestroyDebugUtilsMessengerEXT"));
}

/// <summary>
/// Refresh function is constantly called by QTimer.
/// </summary>
void VulkanWindow::refresh() {
    if (!mStart && mInitialized) {
        mRenderer->setModelMatrix();
        mRenderer->render();
    }
}

/// <summary>
/// ResizeEvent method is called when user adjusts window size.
/// </summary>
/// <param name=""></param>
void VulkanWindow::resizeEvent(QResizeEvent*) {
    // TODO: Implement this.
    qInfo("ResizeEvent is not implemented!");
}

/// <summary>
/// ExposeEvent handler is needed when QWindow is minimized or restored.
/// </summary>
/// <param name=""></param>
void VulkanWindow::exposeEvent(QExposeEvent*) {

    // When Window is restored, rendering must continue.
    if (isExposed() && !mInitialized)
    {
        mInitialized = true;
        //this->initSwapChainResources();
        mRenderer->render();
    }

    // When Window is minimized, rendering must pause.
    if (!isExposed() && mInitialized) {
        vkDeviceWaitIdle(mVulkanPointers.device);
        mInitialized = false;
        //this->releaseSwapChainResources();
    }
}

/// <summary>
/// CloseEvent handler is the right place to delete Vulkan resources. But just in case resources deletion is also 
/// done in QWindow event handler in case QEvent::PlatformSurface.
/// </summary>
/// <param name="e"></param>
void VulkanWindow::closeEvent(QCloseEvent* e) {
    this->releaseSwapChainResources();
    this->releaseResources();
}

/// <summary>
/// QWindow event handler.
/// </summary>
/// <param name="e"></param>
/// <returns></returns>
bool VulkanWindow::event(QEvent* e) {

    if (mRenderer != nullptr && mStart == true) {
        mStart = false;
        this->initSwapChainResources();
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
            this->releaseSwapChainResources();
            this->releaseResources();
        }
        break;
    }

    default:
        break;
    }

    return QWindow::event(e);
}

/// <summary>
/// InitSwapChainResources function creates the parts of Vulkan which may be destroyed and recreated during
/// the application lifetime.
/// </summary>
void VulkanWindow::initSwapChainResources()
{
    qDebug("initSwapChainResources");

    // Get window width and height values.
    const QSize size = this->size() * this->devicePixelRatio();

    // Every window recreate need to set projection matrix in a case user resizes window.
    // We use OpenGL compliant perspective projection parameters.
    float fov = 45.0f;
    float aspectRatio = size.width() / (float)size.height();
    float nearZ = 0.1f;
    float farZ = 1000.0f;
    float top = nearZ * std::tan(M_PI / 180 * fov / 2.0);
    float bottom = -top;
    float right = top * aspectRatio;
    float left = -right;
    QMatrix4x4 proj = QMatrix4x4(2 * nearZ / (right - left), 0, (right + left) / (right - left), 0,
        0, 2 * nearZ / (top - bottom), (top + bottom) / (top - bottom), 0,
        0, 0, -(farZ + nearZ) / (farZ - nearZ), -(2 * farZ * nearZ) / (farZ - nearZ),
        0, 0, -1, 0);

    // Now view is upside down, because Vulkan uses Y-axis down parameters. But we can fix this to Vulkan compliant:
    proj(1, 1) *= -1;
    mRenderer->setProjectionMatrix(proj.data());

    // Then create swap chain.
    if (mRenderer->getSwapChain() != VK_NULL_HANDLE) {
        mRenderer->deleteSwapChain();
    }
    mRenderer->createSwapChain(mVulkanPointers.swapChainSupportDetails, nullptr, nullptr, mRenderer->getSwapChain());
}

void VulkanWindow::releaseSwapChainResources()
{
    qDebug("releaseSwapChainResources");

    // It is important to finish the pending frame right here since this is the
    // last opportunity to act with all resources intact.
    vkDeviceWaitIdle(mVulkanPointers.device);
    mInitialized = false;
    mRenderer->deleteSwapChain();
}


void VulkanWindow::releaseResources()
{
    qDebug("releaseResources");

    // Before releasing resources it is important there aren't any pending work going.
    vkDeviceWaitIdle(mVulkanPointers.device);

    this->releaseSwapChainResources();
    mRenderer->deleteVertexBuffer();
    mRenderer->deleteUniformBuffers();
    mRenderer->deleteSyncObjects();
    mRenderer->deleteCommandPool();
    mRenderer->deleteGraphicsPipeline();
}

/// <summary>
/// This function creates a Vulkan instance. Although Qt offers us Vulkan instance, it doesn't suit for us.
/// </summary>
/// <returns>VkInstance instance.</returns>
VkInstance VulkanWindow::createInstance() {
    qInfo("createInstance");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Simulator";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    std::vector<const char*> reqExtensions = this->getRequiredInstanceExtensions();
    if (enableValidationLayers)
    {

        instInfo.enabledLayerCount = validationLayers.size();
        instInfo.ppEnabledLayerNames = validationLayers.data();

        // When using validation layers, we need VK_EXT_debug_utils extension for debugging,
        // if we want to use custom debug messenger.        
        reqExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        instInfo.enabledExtensionCount = reqExtensions.size();
        instInfo.ppEnabledExtensionNames = reqExtensions.data();
        instInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        instInfo.enabledLayerCount = 0;
        instInfo.enabledExtensionCount = reqExtensions.size();
        instInfo.ppEnabledExtensionNames = reqExtensions.data();
    }

    VkInstance instance;
    if (vkCreateInstance(&instInfo, nullptr, &instance) == VK_SUCCESS) {
        return instance;
    }

    qFatal("Failed to create Vulkan instance!");
    return VK_NULL_HANDLE;
}

/// <summary>
/// This function collects necessary instance extensions. All of these extensions may not be able to 
/// define in compile time, so we define them dynamically.
/// </summary>
/// <returns>Vector containing instance extension names.</returns>
std::vector<const char*> VulkanWindow::getRequiredInstanceExtensions() {
    std::vector<const char*> extensions;
    for (const char* extension : instanceExtensions) {
        if (extension == "__linux__") {
            extensions.push_back(this->getLinuxDisplayType());
        }
        else {
            extensions.push_back(extension);
        }
    }
    return extensions;
}

/// <summary>
/// When running this project in Linux, we need to know what kind of display type we are using. This information
/// cannot be retrieved in compile time, so we need to check it in runtime.
/// </summary>
/// <returns>Vulkan surface extension name for linux.</returns>
const char* VulkanWindow::getLinuxDisplayType() {
    auto env = QProcessEnvironment::systemEnvironment();

    QString value = env.value(QLatin1String("WAYLAND_DISPLAY"));
    if (!value.isEmpty())
        return "VK_KHR_wayland_surface";
    value = env.value(QLatin1String("DISPLAY"));
    if (!value.isEmpty())
        return "VK_KHR_xcb_surface";

    // In case we didn't identify anything, just return extension we need anyways.
    qWarning("Unknown Linux display type");
    return VK_KHR_SURFACE_EXTENSION_NAME;
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
        for (const char* layerName : validationLayers) {
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
        for (const char* layerName : validationLayers) {
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
    auto w = std::make_shared<VulkanWindow>();
    w->resize(1024, 768);
    w->show();

    // If we are running on debug mode, create Vulkan validation layers. 
    QVulkanInstance inst;
    if (enableValidationLayers && !CheckValidationLayerSupport(&inst)) {
        qWarning("Validation layers requested, but not available!");
    }
    else {
        qInfo("Validation layers requested.");
        QList<QByteArray> temp;
        temp.reserve(validationLayers.size());
        for (int i = 0; i < validationLayers.size(); i++) {
            temp.append(validationLayers[i]);
        }
        inst.setLayers(temp);
    }

    // Install needed Vulkan instance extensions.
    QByteArrayList list;
    std::vector<const char*> temp = w->getRequiredInstanceExtensions();
    for (auto extension : temp) {
        list.append(extension);
    }
    inst.setExtensions(list);

    // Assing our own self created Vulkan instance for QVulkanInstance so that it doesn't
    // make its' default instance.
    VkInstance instance = w->createInstance();
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
    w->setupVulkanInstance(inst);

    // Finally start application.
    return app.exec();
}

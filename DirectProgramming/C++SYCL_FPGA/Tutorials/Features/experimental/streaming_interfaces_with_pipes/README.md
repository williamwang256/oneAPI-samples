# `Streaming Interfaces with Pipes` Sample

This FPGA sample is a tutorial that demonstrates how to use pipes to implement and configure a streaming interface on an IP component. If you have not already viewed the [IP Authoring Interfaces Overview Tutorial](), it is recommended that you do so before continuing with this sample.

| Area                  | Description
|:--                    |:--
| What you will learn   | How to use pipes to implement and configure a streaming interface on an IP component
| Time to complete      | 30 minutes
| Category              | Concepts and Functionality

## Purpose

Pipes are a first-in first-out (FIFO) buffer construct that provides links between elements of a design. Access pipes through read and write application programming interfaces (APIs), without the notion of a memory address or pointer to elements within the FIFO.

The concept of a pipe provides us with a mechanism for specifying and configuring streaming interfaces on an IP component. 

## Prerequisites

| Optimized for        | Description
|:---                  |:---
| OS                   | Ubuntu* 18.04/20.04 <br> RHEL*/CentOS* 8 <br> SUSE* 15 <br> Windows* 10
| Hardware             | Intel® Agilex® 7, Arria® 10, and Stratix® 10 FPGAs
| Software             | Intel® oneAPI DPC++/C++ Compiler

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for emulation, generating reports and generating RTL, there are extra software requirements for the simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.

> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

This sample is part of the FPGA code samples. It is categorized as a Tier 2 sample that demonstrates a compiler feature.

```mermaid
flowchart LR
   tier1("Tier 1: Get Started")
   tier2("Tier 2: Explore the Fundamentals")
   tier3("Tier 3: Explore the Advanced Techniques")
   tier4("Tier 4: Explore the Reference Designs")

   tier1 --> tier2 --> tier3 --> tier4

   style tier1 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier2 fill:#f96,stroke:#333,stroke-width:1px,color:#fff
   style tier3 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
   style tier4 fill:#0071c1,stroke:#0071c1,stroke-width:1px,color:#fff
```

Find more information about how to navigate this part of the code samples in the [FPGA top-level README.md](/DirectProgramming/C++SYCL_FPGA/README.md).
You can also find more information about [troubleshooting build errors](/DirectProgramming/C++SYCL_FPGA/README.md#troubleshooting), [running the sample on the Intel® DevCloud](/DirectProgramming/C++SYCL_FPGA/README.md#build-and-run-the-samples-on-intel-devcloud-optional), [using Visual Studio Code with the code samples](/DirectProgramming/C++SYCL_FPGA/README.md#use-visual-studio-code-vs-code-optional), [links to selected documentation](/DirectProgramming/C++SYCL_FPGA/README.md#documentation), and more.


## Key Implementation Details

This tutorial illustrates some key concepts:

- [Configuring a Pipe to Implement a Streaming Interface](#configuring-a-pipe-to-implement-a-streaming-interface)
- [Avalon Streaming Sideband Signals](#avalon-streaming-sideband-signals)
- [Read and Write APIs](#read-and-write-apis)

### Configuring a Pipe to Implement a Streaming Interface

Each individual pipe is a function scope class declaration of the templated `pipe` class.
- The first template parameter is a user-defined type that differentiates this particular pipe from the others and provides a name for the interface in the generated RTL.
- The second template parameter defines the datatype of elements carried by the interface.
- The third template parameter allows you to optionally specify a non-negative integer representing the capacity of the buffer on the input. This can help avoid some amount of bubbles in the pipeline in case the component itself stalls.
- The fourth template parameter uses the oneAPI properties class to allow users to optionally define additional semantic properties for a pipe. **The `protocol` property here is what allows us to configure this pipe to implement a streaming interface.** A list of those properties relevant to this sample is given in Table 1 (please note that this table is *not* complete; see the [FPGA Optimization Guide for Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/docs/oneapi-fpga-add-on/optimization-guide/current/host-pipe-declaration.html) for more information on how to use pipes in other applications).

#### Table 1. Properties to Configure a `pipe` Class to Implement a Streaming Interface

| Property | Valid Values | Default Value |
| ---------| ------------ | ------------- |
| `ready_latency<int>`                    | non-negative integer | 0 |
| `bits_per_symbol<int>`                  | non-negative integer that divides the size of the data type | 8 |
| `uses_valid<bool>`                      | boolean      | `true` |
| `first_symbol_in_high_order_bits<bool>` | boolean      | `true` |
| `protocol`                              | `protocol_avalon_streaming` / `protocol_avalon_streaming_uses_ready` | `protocol_avalon_streaming_uses_ready` |

> **Note:** These properties may be specified in *any* order within the oneAPI `properties` object. Omitting a single property from the properties class instructs the compiler to assume the default value for that property (i.e., you can just define the properties you would like to change from the default). Omitting the properties template parameter entirely instructs the compiler to assume the default values for *all* these properties.

#### Example

The following example declares a pipe to implement a streaming interface using the defaults for all parameters. 

```c++
// Unique user-defined types
class FirstPipeT;

// Pipe properties (listed here are the defaults; this achieves the same
// behavior as not specifying any of these properties)
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>),
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
);

using FirstPipeInstance = sycl::ext::intel::experimental::pipe<
    FirstPipeT,      // An identifier for the pipe
    int,             // The type of data in the pipe
    8,               // The capacity of the pipe
    PipePropertiesT  // Customizable pipe properties
    >;
```

### Avalon Streaming Sideband Signals

You can enable Avalon streaming sideband signal support by using the special `StreamingBeat` struct provided by the `pipes_ext.hpp` header file (`sycl/ext/intel/prototype/pipes_ext.hpp`) as the data type to your pipe. Only the `StreamingBeat` struct generates sideband signals when used with a pipe.

The `StreamingBeat` struct is templated on three parameters.
- The first template parameter defines the type of the data being communicated through the interface.
- The second parameter is used to enable additional 1-bit `start_of_packet` (`sop`) and `end_of_packet` (`eop`) signals to the Avalon interface.
- The third template parameter is used to enable the `empty` signal, which indicates the number of symbols that are empty during the `eop` cycle.

#### Example

The following example shows how to configure a `StreamingBeat` struct with the `sop`, `eop`, and `empty` signals (the second and third template parameters are set to `true`). 

```c++
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

...

using StreamingBeatDataT = sycl::ext::intel::experimental::StreamingBeat<unsigned char, true, true>;
```

> **Note**: The size of the datatype carried by the interface must be a multiple of the `bits_per_symbol` property associated with the pipe (the default is 8).

### Read and Write APIs

The read and write APIs are the same as for all pipes. See the [IP Authoring Interfaces Overview Tutorial]() for more information.

### Testing the Tutorial

In `threshold_packets.cpp`, two pipes are declared for implementing the streaming input and streaming output interfaces on the Threshold kernel, which thresholds pixel values in an image. The streams use start of packet and end of packet signals to determine the beginning and end of the image.


#### Reading the Reports

1. After compiling in the reports flow, locate and open the `report.html` file in the `threshold_packets.report.prj/reports/` directory.
2. Open the **Views** menu and select **System Viewer**.
3. In the left-hand pane, select **Threshold** under the System hierarchy.

In the main **System Viewer** pane, the streaming in and streaming out interfaces are shown by the pipe read and pipe write nodes respectively. Clicking on either of these nodes gives further information for these interfaces in the **Details** pane. This pane will show that the read is reading from `InPixel`, and that the write is writing to `OutPixel`, as well as verifying that both interfaces have a width of 24 bits (corresponding to the `StreamingBeatT` type) and depth of 8 (which is the capacity that each pipe was declared with).

<p align="center">
  <img src=assets/kernel.png />
</p>

#### Viewing the Simulation Waveforms

1. After compiling and running in the simulation flow, locate and run the `view_waveforms.sh` script in the `threshold_packets.fpga_sim.prj/` directory.



## Build the `Streaming Interfaces with Pipes` Tutorial

>**Note**: When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script in the root of your oneAPI installation every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files(x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### On Linux*

1. Change to the sample directory.
2. Build the program for Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      make fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      make report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      make fpga_sim
      ```
   4. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      make fpga
      ```	

### On Windows*

1. Change to the sample directory.
2. Build the program for the Intel® Agilex® 7 device family, which is the default.
   ```
   mkdir build
   cd build
   cmake -G "NMake Makefiles" ..
   ```
   > **Note**: You can change the default target by using the command:
   >  ```
   >  cmake -G "NMake Makefiles" .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
   >  ```
   >
   > This tutorial only uses the IP Authoring flow and does not support targeting an explicit FPGA board variant and BSP.

3. Compile the design. (The provided targets match the recommended development flow.)

   1. Compile and run for emulation (fast compile time, targets emulated FPGA device).
      ```
      nmake fpga_emu
      ```
   2. Generate the optimization report. (See [Read the Reports](#read-the-reports) below for information on finding and understanding the reports.)
      ```
      nmake report
      ```
   3. Compile and run for simulation (fast compile time, targets simulated FPGA device).
      ```
      nmake fpga_sim
      ```
   4. Compile and run for FPGA hardware (longer compile time, targets an FPGA device).
      ```
      nmake fpga
      ```
> **Note**: If you encounter any issues with long paths when compiling under Windows*, you may have to create your ‘build’ directory in a shorter path, for example c:\samples\build.  You can then run cmake from that directory, and provide cmake with the full path to your sample directory.

## Run the `Streaming Interfaces with Pipes` Tutorial

### On Linux

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   ./host_pipes.fpga_emu
   ```
2. Run the sample on the FPGA simulator.
   ```
   CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1 ./host_pipes.fpga_sim
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.
	
### On Windows

1. Run the sample on the FPGA emulator (the kernel executes on the CPU).
   ```
   host_pipes.fpga_emu.exe
   ```
2. Run the sample on the FPGA simulator.
   ```
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=1
   host_pipes.fpga_sim.exe
   set CL_CONTEXT_MPSIM_DEVICE_INTELFPGA=
   ```
> **Note**: Running this sample on an actual FPGA device requires a BSP that supports host pipes. As there are currently no commercial BSPs with such support, only the IP Authoring flow is enabled for this code sample.

## Example Output

```
Data:  0
Data:  1
Data:  2
Data:  5
Data:  8
Data: 11
Data: 14
Data: 18
Data: 22
Data: 27
Data: 31
Data: 36
Data: 41
Data: 46
Data: 52
Data: 58

PASSED
```

## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

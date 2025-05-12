package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.Arg2pSolverFactory
import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.argumentation.core.libs.basic.FlagsBuilder
import it.unibo.tuprolog.core.Clause
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.theory.Theory
import org.queueinc.hamlet.argumentation.SpaceGenerator
import org.queueinc.hamlet.argumentation.SpaceMining
import org.queueinc.hamlet.automl.*
import kotlin.random.Random


class Controller(private val debugMode: Boolean, private val dataManager: FileSystemManager) {

    private var lastSolver : MutableSolver? = null
    private var theory = ""

    private var generationTime = 0L

    private lateinit var config: Config

    private fun nextIteration() = config.copy(iteration = config.iteration + 1)
    private fun updateIteration() {
        config.iteration++
        dataManager.saveConfig(config)
    }

    fun init(mode: String, batchSize: Int, timeBudget: Int, seed: Int) {
        // stopAutoML()
        // runAutoML(dataManager.workspacePath, dataManager.volume, debugMode)
        config = dataManager.loadConfig().let {
            if (it == null) Config(0, null, null, null, emptyList(), mode, batchSize, timeBudget, seed)
            // {
            //    dataManager.cleanWorkspace()
            //    dataManager.initWorkspace()
            //    Config(0, null, null, null, emptyList(), mode, batchSize, timeBudget, seed)
            // }
            else it.copy(mode = mode, batchSize = batchSize, timeBudget = timeBudget, seed = seed)
        }
    }

    fun stop() {
        // stopAutoML()
    }

    fun knowledgeBase() : String? = dataManager.loadKnowledgeBase(config.copy())

    fun autoMLData() : AutoMLResults? = dataManager.loadAutoMLData(config.copy())

    fun graphData() : MutableSolver? =
        dataManager.loadGraphData(config.copy())?.let { graph ->
            theory = dataManager.loadKnowledgeBase(config.copy()) ?: ""
            Arg2pSolverFactory.default(
                staticLibs = listOf(SpaceGenerator),
                dynamicLibs = listOf(SpaceMining),
                settings = FlagsBuilder(
                    argumentLabellingMode = "grounded_hash",
                    graphExtensions = listOf("standardPref"),
                    orderingPrinciple = "last",
                    orderingComparator = "democrat"
                ).create(),
                theory = theory
            ).also {
                lastSolver = it
                arg2pScope {
                    it.solve("miner" call "load_graph_data"(Struct.parse(graph))).first()
                }
            }
        }

    fun generateGraph(theory: String, blocking: Boolean, update: (MutableSolver) -> Unit) {
        this.theory = theory
        val start = System.currentTimeMillis() / 1000
        val creationRules = SpaceGenerator.createGeneratorRules(theory)
        val solver = Arg2pSolverFactory.default(
            staticLibs = listOf(SpaceGenerator),
            dynamicLibs = listOf(SpaceMining),
            settings = FlagsBuilder(
                graphExtensions = listOf("standardPref"),
                orderingPrinciple = "last",
                orderingComparator = "democrat"
            ).create(),
            theory = theory + "\n" + creationRules
        ).also { solver ->
            arg2pScope {
                solver.solve("prepare_sensitive_groups"(Y)).map { it ->
                    val b = it.substitution[Y] ?.castToList()?.toList()?.map { x -> Clause.of(x.castToStruct()) } ?: emptyList()
                    solver.appendStaticKb(Theory.of(b))
                    println("Sensitive groups ready!")
                }.first()
                solver.solve("prepare_theory"(X)).map {
                    val a = it.substitution[X] ?.castToList()?.toList()?.map { x -> Clause.of(x.castToStruct()) } ?: emptyList()
                    solver.appendStaticKb(Theory.of(a))
                    println("Theory ready!")
                }.first()
                println("Theory:")
                println(solver.staticKb.toString(asPrologText = true))
            }
        }

        Thread {
            solver.solve(
                Struct.parse("buildLabelSetsSilent"),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)
            ).first()

            update(solver)
            lastSolver = solver
            generationTime = (System.currentTimeMillis() / 1000) - start
        }.also {
            it.start()
            if (blocking) it.join()
        }
    }

    fun launchAutoML(blocking: Boolean, update: (AutoMLResults) -> Unit) {
        lastSolver?.also { solver ->
            Thread {

                println("Checking Config")
                config = SpaceTranslator.mineConfig(config.copy(), solver).also {
                    if (it.dataset != config.dataset) {
                        dataManager.cleanWorkspace()
                        dataManager.initWorkspace()
                        it.iteration = 0
                    }
                }

                if (config.dataset == null || config.metric == null || config.fairnessMetric == null || config.sensitiveFeatures.isEmpty()) {
                    println("Missing Optimization Data")
                    return@Thread
                }

                println("Saving Graph")
                dataManager.saveKnowledgeBase(nextIteration(), this.theory)
                dataManager.saveGraphData(nextIteration(), dumpGraphData() ?: "")

                println("Exporting AutoML input")
                dataManager.saveAutoMLData(nextIteration(), generationTime, solver)

                println("Input created for iteration ${nextIteration()}")

                execAutoML(dataManager.workspacePath, nextIteration(), debugMode)

                if (dataManager.existsAutoMLData(nextIteration())) {
                    updateIteration()
                    val res = dataManager.loadAutoMLData(config.copy())!!
                    dataManager.saveGeneratedRules(config.copy(),
                        res.inferredRules.distinctBy { it.theoryRepresentation }
                            .joinToString("\n") { it.theory }
                    )
                    update(res)
                } else {
                    // input.delete()
                }
            }.also {
                it.start()
                if (blocking) { it.join() }
            }
        }
    }

    private fun dumpGraphData() : String? =
        arg2pScope {
            lastSolver?.solve("miner" call "dump_graph_data"(X))
                ?.filter { it.isYes }
                ?.map { it.substitution[X].toString() }
                ?.first()
        }
}